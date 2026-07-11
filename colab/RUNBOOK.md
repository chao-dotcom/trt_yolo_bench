# Colab runbook

Cell-by-cell plan for running this project on Google Colab's free T4 GPU. This
is the recommended environment per `PLAN.md` §0 — write and review code
locally (no GPU needed for that), but every cell that produces a number in
`RESULTS.md` must actually execute here (or on an equivalent CUDA box).

Runtime → change runtime type → **T4 GPU** before starting.

## 1. Confirm the GPU and CUDA version

```bash
!nvidia-smi
!nvcc --version
```

Note the CUDA version printed — it determines which TensorRT package version
to install in step 3.

## 2. Clone this repo

```bash
!git clone https://github.com/chao-dotcom/trt_yolo_bench.git
%cd trt_yolo_bench
```

## 3. Install TensorRT — the one real risk, per `PLAN.md` §10

Two things are needed, not one: the **Python bindings** (`pip install
tensorrt` is enough for this) and the **C++ SDK** (headers like `NvInfer.h` +
`libnvinfer.so`, which the pip wheel does *not* reliably ship). For the C++
harness, follow NVIDIA's documented Debian/apt install matching the CUDA
version from step 1 (the `cuda-keyring` repo method) — this is the standard
path on an apt-based system like Colab's Ubuntu image, and gives root access
to install system packages.

Verify before writing any C++ against it:

```bash
!dpkg -l | grep -i nvinfer
!find / -name "NvInfer.h" 2>/dev/null
```

If this genuinely doesn't resolve after reasonable effort, fall back to a
rented GPU VM per `PLAN.md` §0 — don't sink hours into a Colab-specific apt
quirk that a plain Ubuntu box wouldn't have.

## 4. Python dependencies

```bash
!pip install -q ultralytics onnx pycuda pycocotools requests
```

(`tensorrt` already installed in step 3.)

## 5. Data slice

```bash
!python export/download_coco_slice.py
```

Downloads the two COCO annotation files, extracts `instances_val2017.json` /
`instances_train2017.json`, and fetches the ~1200 individual images for the
val (1000) + calibration (200) slices — see `PLAN.md` §3 for exactly which
images these are (deterministic, checked into `data/val_slice.json` /
`data/calib_slice.json`).

## 6. Export + build engines

```bash
!python export/export_onnx.py
!python export/build_engines.py   # produces engines/yolo11n_{fp32,fp16,int8}.engine
```

INT8 build reads `data/calib_slice.json`, writes/reuses `calib.cache`.

## 7. Build the C++ harness

```bash
%cd harness
!mkdir -p build && cd build && cmake .. && make -j
%cd ../..
```

## 8. Run the benchmark, per precision

```bash
!./harness/build/trt_yolo_bench engines/yolo11n_fp32.engine data/val_slice.json --out results_fp32.json
!./harness/build/trt_yolo_bench engines/yolo11n_fp16.engine data/val_slice.json --out results_fp16.json
!./harness/build/trt_yolo_bench engines/yolo11n_int8.engine data/val_slice.json --out results_int8.json
```

Each run prints the latency/throughput stats to stdout (copy verbatim into
`RESULTS.md`) and writes a COCO-format detections JSON.

## 9. Score accuracy

```bash
!python eval/score_coco.py --dets results_fp32.json
!python eval/score_coco.py --dets results_fp16.json
!python eval/score_coco.py --dets results_int8.json
```

Copy each `COCOeval.summarize()` table verbatim into `RESULTS.md`.

## 10. Fill in `RESULTS.md` and push

Update the environment section (GPU/driver/CUDA/TensorRT versions from step
1+3), the headline table, and both verbatim-output sections. Commit from
Colab or copy the numbers back to a normal git checkout — either way, don't
transcribe by hand into a *different* summarized form; paste the real output.
