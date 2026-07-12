# trt_yolo_bench

**YOLO11 → TensorRT in hand-written C++.** FP32, FP16, INT8 precision comparison with custom INT8 calibrator. Accuracy on COCO val2017, latency benchmarked end-to-end with no black-box postprocessing.

---

## Results

**Status:** ⏳ Implementation complete, GPU run pending.  
See [`RESULTS.md`](RESULTS.md) for the full output once benchmarked on real NVIDIA GPU.

```
Precision Tradeoff Matrix (schema — populated after GPU run)

                    FP32          FP16          INT8
  ───────────────────────────────────────────────────
  Latency (ms)      [====]        [==]          [=]
  mAP50 (%)         [========]    [========]    [====]
  Model Size (MB)   [========]    [====]        [==]
  
  Target: sub-10ms @ 640x640 with < 2% accuracy loss
```

**What you'll get:** Measured latency, accuracy, memory footprint across three precisions on a real GPU.

---

## Quick Start

### Download & export
```bash
python export/download_coco_slice.py
python export/export_onnx.py --weights yolo11n.pt --out-dir models
python export/build_engines.py --onnx models/yolo11n.onnx --engines-dir engines
```

### Benchmark (C++)
```bash
cd harness
cmake -S . -B build && cmake --build build --config Release
./build/trt_yolo_bench ../engines/yolo11n_fp16.engine ../data/val_slice.json --out ../results_fp16.json
```

### Evaluate
```bash
python eval/score_coco.py --dets results_fp16.json
```

---

## What's Actually Here

- **`export/`** — PyTorch → ONNX → TensorRT engines (FP32/FP16/INT8)
  - Custom INT8 entropy calibrator (not torch quantization)
  - No Ultralytics export wrapper
  
- **`harness/`** — C++ inference loop
  - Direct `nvinfer1` API (CUDA buffers, letterbox preprocessing, manual NMS)
  - Hand-decoded YOLO detection head (`[1, 84, 8400]` → detections)
  - Latency timer on the critical path
  
- **`eval/`** — COCO val2017 scoring
  - Official `pycocotools` evaluator
  - Matches reference methodology
  
- **`colab/`** — GPU runner (author on CPU machine; results will be generated here)

---

## Why This Repo

This isn't "I called `model.export()`" — it's reproducible engineering evidence:
ownership of the export pipeline, INT8 calibration tradeoffs, and real C++ deployment
all testable against COCO val2017 ground truth. See [`PLAN.md`](PLAN.md) for design choices.

---

## The Harness

Expects YOLO-style output at 640×640: `[1, 84, 8400]` tensors, writes COCO results JSON. No external NMS or postprocessing libraries — all hand-written in the detection decode stage.
