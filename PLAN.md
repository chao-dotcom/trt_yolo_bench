# PLAN.md — build plan for `trt_yolo_bench`

This doc is written for whoever implements this next (agent or human) and has
**no other context**. Read this whole file before writing any code — several
decisions below exist specifically to prevent redoing work or producing numbers
that can't be defended.

---

## §0. STOP AND READ FIRST — this cannot be built or run on the authoring machine

The machine this plan was written on has **no NVIDIA GPU** (Intel UHD Graphics
only, confirmed via `nvidia-smi` absence and `Get-CimInstance Win32_VideoController`).
TensorRT is NVIDIA-only — there is no CPU fallback and no way to produce a real
`.engine` file or a real latency number without an actual NVIDIA GPU + driver.

**Do not fabricate numbers to work around this.** That is the one hard rule in
every project in this author's portfolio (see `RESULTS.md` for the honesty
format). If you are an agent picking this up and you don't have GPU access
either, the correct move is to write and unit-test everything that *can* run
without a GPU (Python export scripts up to the ONNX step, C++ code compiled
with stubbed/mocked TensorRT calls if needed, the scoring script logic against
synthetic detections) and leave `RESULTS.md` untouched — not to guess plausible
numbers.

**Recommended environment, in priority order:**
1. **Google Colab, free tier (T4 GPU)** — this author has already used Colab
   for a prior GPU-dependent feasibility experiment in this job search, so
   there's working precedent. Free tier is sufficient: this project needs no
   training, only inference + a short INT8 calibration pass on ~200 images.
   Colab gives root-ish access (`!apt`, `!pip`, shell cells), CUDA + driver
   preinstalled — but **TensorRT with C++ headers is not preinstalled** and
   must be installed explicitly (see §5, this is the one real risk in the
   whole plan — budget time for it).
2. **A rented on-demand GPU VM** (Lambda Labs / RunPod / Paperspace, cheapest
   single NVIDIA GPU instance) — fallback only if Colab's TensorRT C++ install
   proves genuinely blocked (e.g. driver/CUDA version mismatch that `apt`
   can't resolve). Full root access makes the NVIDIA-documented Debian install
   path for TensorRT reliable. Costs real money — check with the project owner
   before spinning one up.

Everything in `export/`, `harness/`, `eval/` can be **written and code-reviewed**
on a CPU-only machine. It can only be **run for real numbers** on a CUDA box.
Structure the work so that's the only thing blocked on GPU access, not the
whole project.

---

## §1. Goal, in one sentence

Take a real, named, modern object detector to TensorRT at three precisions
using the actual Builder/calibration API (not a wrapper's one-liner), drive it
from a hand-written C++ inference program, and report honest latency/
throughput/accuracy-retention numbers scored against real COCO ground truth.

## §2. Scope decisions already made — do not re-litigate these without a reason

| Decision | Choice | Why |
|---|---|---|
| Detector | **YOLO11n** (Ultralytics, COCO-pretrained) | Smallest/fastest variant, cleanest ONNX export path, well-documented output format for hand-decoding |
| Fallback detector | YOLOv8n | Only if YOLO11's export graph hits an ONNX/TensorRT op incompatibility not worth debugging — same output format, same decode logic |
| Precisions | **FP32, FP16, INT8** | The three that TensorRT natively supports for this model without extra plugins |
| Engine building | **TensorRT Builder Python API** (`tensorrt` package: `trt.Builder`, `trt.OnnxParser`, `trt.IBuilderConfig`) directly, not `ultralytics export(format='engine')` | A one-line export doesn't produce anything defensible in an interview. Building against the Builder API means the precision flags, workspace size, and (for INT8) the calibrator are things the author actually wrote and can explain. |
| INT8 calibration | **Custom `trt.IInt8EntropyCalibrator2` subclass**, Python, reading a fixed 200-image calibration set | This is the single highest-value piece of "real TensorRT experience" in the whole project — do not use a library's auto-calibration shortcut |
| Inference + benchmark harness | **C++, `nvinfer1` directly** (`IRuntime::deserializeCudaEngine`, `IExecutionContext`) | This is what proves C++ deployment ability — see §4 |
| Preprocessing | Hand-written **letterbox resize + normalize** in the C++ harness (OpenCV allowed for image I/O and the resize call itself, but the letterbox math — scale, padding, coordinate bookkeeping — must be explicit code, not a library call) | |
| Postprocessing | Hand-written **YOLO head decode + NMS** in C++, no TensorRT NMS plugin, no Ultralytics postprocessing code | This is the second highest-value piece of "real C++ algorithm work" — see §4.3 |
| Accuracy scoring | **`pycocotools.cocoeval.COCOeval`** (reference library) scoring the C++ harness's own output dumped to COCO-format JSON | Mirrors this author's existing pattern in other projects: write the core logic yourself, validate against a trusted reference library — never hand-roll the metric that becomes the headline number |
| Dataset | **COCO val2017**, a fixed deterministic 1000-image slice for benchmarking/scoring; a separate fixed 200-image slice from COCO train2017 for INT8 calibration | Real, recognized benchmark; slices keep download size small; see §3 for exact selection method |
| Explicitly OUT of scope | SAM, ViT (as standalone claims) | SAM is segmentation, a different task — don't blur it into a detection benchmark. ViT is too generic to point at a concrete result here. Neither gets mentioned in any output of this project. |
| Optional stretch | **RT-DETR** (Ultralytics-supported transformer detector) run through the *exact same* export → TensorRT → C++ harness pipeline | Only attempt after YOLO11n numbers are fully real and in `RESULTS.md`. If done, it honestly extends the claim to "CNN and transformer-based detectors." If not done, say nothing about it anywhere public. |
| License note | Ultralytics code is **AGPL-3.0** | Fine for this use (personal benchmark/portfolio artifact, source available, not a redistributed closed product). Would need a commercial Ultralytics license if this ever became part of a closed-source product — not relevant here, just noted so it isn't a surprise later. |

## §3. Dataset — exact, reproducible slice selection

No random seeds — fully deterministic by construction, so re-running this plan
five years from now on a different machine with different library versions
still produces the same slice.

1. **Annotations**: download `annotations_trainval2017.zip` from
   `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`
   (one-time, ~241 MB). Extract only `instances_val2017.json` and
   `instances_train2017.json`; delete the rest (captions/person_keypoints files
   not needed).
2. **Benchmark/mAP slice (val)**: from `instances_val2017.json`, take the
   `images` list, sort by `id` ascending, take the **first 1000 image
   entries**. Save the resulting list of `{id, file_name}` to
   `data/val_slice.json` (check this file into git — it's small and makes the
   slice reviewable/reproducible without re-deriving it).
3. **Calibration slice (train)**: from `instances_train2017.json`, same
   method — sort by `id` ascending, take the **first 200 image entries**, save
   to `data/calib_slice.json`. Train/val are disjoint sets by construction, so
   no overlap with the benchmark slice is possible.
4. **Image download**: for each entry in both slice files, fetch the image
   directly from COCO's CDN — `http://images.cocodataset.org/val2017/{file_name}`
   or `.../train2017/{file_name}` — one file at a time. Do **not** download the
   full `val2017.zip` (778 MB) or `train2017.zip` (18 GB); 1200 individual
   image fetches is fast and avoids the multi-GB downloads entirely. Write this
   as `export/download_coco_slice.py`, idempotent (skip files already on disk).
5. `data/` is gitignored — never commit images or the 241 MB annotations zip,
   only the two small `*_slice.json` files.

## §4. The C++ harness — what "hand-written" means here, precisely

This is the part that actually proves something, so be exact about it.

### 4.1 Engine loading
- `nvinfer1::IRuntime::deserializeCudaEngine()` on the `.engine` file produced
  by the Python export step (§5).
- Explicit CUDA buffer allocation (`cudaMalloc`) for input/output bindings,
  explicit host↔device copies (`cudaMemcpyAsync`) — not a convenience wrapper.
- One `IExecutionContext` reused across the whole benchmark loop (don't
  recreate it per image — that would pollute the latency numbers).

### 4.2 Preprocessing (hand-written)
- Load image (OpenCV `imread` is fine — decoding JPEG by hand is not the
  point of this project).
- **Letterbox resize to 640×640**: compute `scale = min(640/w, 640/h)`, resize
  to `(w*scale, h*scale)`, pad the remainder with constant value 114 (gray) to
  reach 640×640, keeping the resized image centered or top-left (pick one, be
  consistent, and store the exact `(scale, pad_x, pad_y)` per image — needed
  to map detections back to original image coordinates for scoring).
- Normalize to `[0,1]`, HWC→CHW, BGR→RGB (Ultralytics models expect RGB).
- This math must be explicit in the C++ code — OpenCV may do the resize call,
  but the letterbox scale/pad bookkeeping is the harness's own logic.

### 4.3 Postprocessing (hand-written — the core deliverable)
YOLO11/YOLOv8 ONNX export (no NMS baked in) produces one output tensor shaped
`[1, 4+80, 8400]` for COCO's 80 classes at 640 input resolution (8400 =
sum of the three detection-head strides' grid cells). Decode steps:
1. Transpose to `[8400, 84]`.
2. Split into box `[:, :4]` (cx, cy, w, h, in **model input pixel space**,
   i.e. relative to the 640×640 letterboxed image) and class scores `[:, 4:]`.
3. Per row: take `max` over the 80 class scores as confidence + its index as
   class id. Filter rows below a confidence threshold (start at 0.25).
4. Convert `(cx, cy, w, h)` → `(x1, y1, x2, y2)`, then undo the letterbox
   transform using the stored `(scale, pad_x, pad_y)` from §4.2 to get boxes in
   **original image pixel coordinates** — this step is required for COCO
   scoring to be meaningful.
5. **Greedy NMS, hand-written**: sort surviving boxes by confidence
   descending; iterate, keep a box, suppress any remaining box with IoU > 0.45
   against it (per class — don't suppress across different classes). No
   TensorRT NMS plugin, no `cv::dnn::NMSBoxes`. This is the one piece of
   algorithmic code in the harness that must not come from a library call, the
   same way `mot_tracker`'s IoU/association code is hand-written even though
   `scipy` is used for the assignment solver underneath it.

### 4.4 Benchmark timing
- Warm up 20 iterations (excluded from timing) before measuring.
- Time **300+ iterations** per engine (FP32/FP16/INT8 separately), wall-clock
  around `context->enqueueV3()` (or `executeV3`, whichever this TensorRT
  version exposes) + the device sync, **not** including image file I/O.
- Report mean, p50, p99 latency (ms) and derived throughput (FPS = 1000/mean).
- Also record engine file size on disk (`ls -la *.engine`) as the model-size
  metric per precision.

### 4.5 Output for scoring
- Dump all detections across the 1000-image benchmark slice to a single COCO-
  format results JSON (`[{"image_id":..., "category_id":..., "bbox":[x,y,w,h],
  "score":...}, ...]`) — this is the standard format `pycocotools` expects.
- One results JSON per precision (`results_fp32.json`, `results_fp16.json`,
  `results_int8.json`).

## §5. Export / engine-building pipeline (Python, `export/`)

1. `export_onnx.py` — load YOLO11n via `ultralytics.YOLO('yolo11n.pt')`,
   `model.export(format='onnx', opset=17, dynamic=False, imgsz=640)`. Verify
   the ONNX output shape matches `[1, 84, 8400]` before moving on — if it
   doesn't, the decode logic in §4.3 needs to change to match, don't guess.
2. `calibrator.py` — a class implementing `trt.IInt8EntropyCalibrator2`:
   `get_batch()` reads and preprocesses (same letterbox logic as §4.2, ported
   to Python — keep the two implementations consistent) a batch from the
   200-image calibration slice; `read_calibration_cache()` /
   `write_calibration_cache()` persist the cache to `calib.cache` so
   calibration doesn't rerun on every build.
3. `build_engines.py` — for each precision:
   - **FP32**: `trt.Builder` + `OnnxParser`, no special flags, `build_engine()`.
   - **FP16**: same, set `config.set_flag(trt.BuilderFlag.FP16)`.
   - **INT8**: set `config.set_flag(trt.BuilderFlag.INT8)`,
     `config.int8_calibrator = <calibrator.py instance>`.
   - Serialize each to `engines/yolo11n_{fp32,fp16,int8}.engine`.
4. Sanity check before trusting any downstream number: run each engine
   through the **Python** TensorRT runtime (`pycuda` or `cuda-python`) on a
   single test image first and visually/manually confirm the decoded boxes
   look right (e.g. dump an annotated image) — catching a decode bug here is
   far cheaper than debugging it after a 1000-image C++ benchmark run produces
   a suspiciously bad mAP number.

## §6. Scoring (`eval/score_coco.py`)

- Load `instances_val2017.json`, restrict to the 1000 image ids in
  `data/val_slice.json` (`cocoGt.getImgIds()` filtered, or construct a
  sub-COCO with just those ids).
- `cocoDt = cocoGt.loadRes('results_fp32.json')` (repeat per precision).
- `COCOeval(cocoGt, cocoDt, 'bbox')` → `.evaluate()` → `.accumulate()` →
  `.summarize()` — this prints the standard 12-metric COCO table
  (AP@[.5:.95], AP@.5, AP@.75, AP across small/medium/large, AR variants).
  The headline number for `RESULTS.md` is **AP@[.5:.95]** (what everyone means
  by "mAP" without qualification) plus AP@.5 as a secondary, more forgiving
  number.
- Run this once per precision, keep the full printed table for `RESULTS.md`
  (verbatim output, per this author's existing convention — see §7).

## §7. `RESULTS.md` rules — read before writing a single number

This repo follows the same honesty convention as this author's other
benchmark projects:
- **Every number must come from actually running the code.** No "expected"
  or "typical" numbers standing in for a real run, even temporarily.
- State the exact environment (GPU model, driver version, CUDA version,
  TensorRT version, OS) — TensorRT numbers are not portable across driver/
  version combinations and an interviewer may ask.
- State the exact command used to produce each number.
- Include verbatim tool output (the `COCOeval.summarize()` table, the
  benchmark harness's printed latency stats) — not a hand-transcribed summary
  that could silently introduce an error.
- If INT8 accuracy drops more than expected, **say so and explain why**
  (calibration set too small/unrepresentative, a class that's rare in the
  200-image calibration slice, etc.) — an honest, explained regression is more
  credible than a suspiciously perfect INT8 number.
- Final table shape:

  | Precision | AP@[.5:.95] | AP@.5 | Mean latency (ms) | p99 (ms) | Throughput (FPS) | Engine size (MB) |
  |---|---|---|---|---|---|---|
  | FP32 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
  | FP16 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
  | INT8 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

  Do not fill in a row until every cell in it is a real measured value.

## §8. Draft resume bullet (do not use until §7's table is fully real)

> Exported YOLO11n (PyTorch → ONNX → TensorRT) at FP32/FP16/INT8 via the
> TensorRT Builder API with a custom INT8 entropy calibrator; built a **C++
> (nvinfer1) inference harness** with hand-written letterbox preprocessing and
> YOLO-head decode + NMS — measured **[X]× speedup at [Y]% mAP retention**
> (COCO val, `pycocotools`-scored) across precisions.

Fill in `[X]` and `[Y]` only from the real `RESULTS.md` table.

## §9. Definition of done

- [ ] `export/` runs end-to-end on a CUDA box: PyTorch → ONNX → 3 engines
- [ ] `harness/` compiles against a real TensorRT install and runs all 3 engines
- [ ] `eval/score_coco.py` produces a real `COCOeval` table for all 3 precisions
- [ ] `RESULTS.md` fully filled in, verbatim outputs included, environment stated
- [ ] README's "Status" line updated from "planning complete, implementation
      pending" to reflect reality
- [ ] (stretch, optional) RT-DETR run through the same pipeline, its own row
      added to `RESULTS.md`, clearly labeled as a second architecture

## §10. Known risks, ranked

1. **TensorRT C++ SDK install on Colab** (headers + `libnvinfer.so`, not just
   the Python bindings) — budget real time for this, it's the least
   standardized step. NVIDIA's documented path is the `cuda-keyring` apt repo
   method matching Colab's CUDA version; verify with
   `dpkg -l | grep libnvinfer-dev` and confirm `NvInfer.h` exists somewhere
   under `/usr/include` before writing any C++ against it.
2. **ONNX export op incompatibility** — YOLO11's export graph should convert
   cleanly (Ultralytics tests this), but if the TensorRT `OnnxParser` rejects
   an op, don't hand-patch the ONNX graph — fall back to YOLOv8n (identical
   decode logic, more battle-tested export path) rather than sinking time into
   a graph-surgery detour that isn't the point of this project.
3. **INT8 accuracy collapse** — if AP drops drastically (not just "a bit
   worse than FP16," but collapsed), the likely cause is calibration-batch
   preprocessing not matching the harness's own preprocessing exactly (see
   §5.2's note to keep the two implementations consistent) — check that first
   before concluding INT8 "doesn't work" for this model.
