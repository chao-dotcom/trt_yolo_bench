# trt_yolo_bench

**YOLO11 → TensorRT inference harness in hand-written C++.**

End-to-end precision benchmarking (FP32, FP16, INT8) with custom INT8 entropy calibrator, direct `nvinfer1` C++ runtime, and COCO val2017 accuracy evaluation. No wrapper libraries, no black-box postprocessing.

---

## Core Components

**Export Pipeline** (`export/`)
- PyTorch → ONNX → TensorRT engine builder
- Custom INT8 entropy calibrator with calibration dataset
- Outputs FP32 / FP16 / INT8 engines ready for inference

**Inference Harness** (`harness/`)
- Direct `nvinfer1` C++ API
- CUDA buffer management, letterbox preprocessing
- Hand-decoded YOLO detection head: `[1, 84, 8400]` → bounding boxes + confidence
- Custom NMS (no TensorRT plugin dependency)
- Latency measurement on critical path

**Evaluation** (`eval/`)
- Benchmark output scored against COCO val2017 ground truth
- Official `pycocotools` reference evaluator
- Outputs mAP50, mAP75, per-class metrics

---

## Usage

```bash
# Export YOLO11 to TensorRT engines
python export/download_coco_slice.py
python export/export_onnx.py --weights yolo11n.pt --out-dir models
python export/build_engines.py --onnx models/yolo11n.onnx --engines-dir engines

# Run inference and benchmark
cd harness
cmake -S . -B build && cmake --build build --config Release
./build/trt_yolo_bench ../engines/yolo11n_fp16.engine ../data/val_slice.json --out results.json

# Evaluate accuracy
python eval/score_coco.py --dets results.json
```
