# trt_yolo_bench — YOLO → TensorRT, benchmarked from a hand-written C++ harness

A from-scratch **C++ TensorRT inference harness** (`nvinfer1`, no Python in the
hot path) for a YOLO object detector exported at **FP32 / FP16 / INT8**, with a
**custom-written INT8 calibrator**, **hand-decoded detection head + NMS** (no
TensorRT NMS plugin, no Ultralytics postprocessing), and accuracy measured on
**real COCO val2017** with the reference `pycocotools` evaluator.

The point of this repo is not "I called `model.export(format='engine')`" — it's
a real answer to "show me a model you took to silicon and the tradeoff you
owned": what changes at each precision, why, and by how much.

> Status: **planning complete, implementation pending.** See [`PLAN.md`](PLAN.md)
> for the full build plan. [`RESULTS.md`](RESULTS.md) is a placeholder — every
> number in it will say `PENDING` until produced by actually running this code.
> No numbers are invented ahead of the run, ever — see the Honesty rules in
> `PLAN.md` §7.

---

## Why this exists

TensorRT and ONNX are easy to *list* and hard to *prove*. This repo is a single
coherent artifact that produces real, reproducible engineering evidence:

1. **TensorRT competency** — engines built directly against the TensorRT
   Builder API (not a one-line `ultralytics` export), including a hand-written
   INT8 entropy calibrator, so the precision/calibration tradeoffs are
   something the author can actually defend in an interview.
2. **A named, modern detection architecture** — YOLO11 (Ultralytics), with an
   optional RT-DETR (transformer-based) comparison if time allows — see
   `PLAN.md` §2 for exactly what is and isn't in scope.
3. **Real C++ deployment evidence** — the inference harness that produces the
   latency/throughput numbers is C++ against `nvinfer1` directly: manual CUDA
   buffer management, hand-written letterbox preprocessing, and a hand-written
   YOLO head decode + NMS. No black-box postprocessing.

## What's planned (see `PLAN.md` for the authoritative spec)

```
trt_yolo_bench/
  export/            Python: PyTorch -> ONNX, ONNX -> TensorRT engines (FP32/FP16/INT8 + calibrator)
  harness/            C++: nvinfer1 runtime, preprocessing, detection decode + NMS, benchmark timer
  eval/               Score harness output against COCO val2017 ground truth via pycocotools
  data/               (gitignored) COCO image slice + annotations, fetched by a download script
  colab/              GPU runbook — this repo is authored on a machine with no NVIDIA GPU (see PLAN.md §0)
  RESULTS.md          Real numbers only, once measured
```

## Honesty note

Every number that will ever appear in `RESULTS.md` comes from **running the
code in this repo on a real NVIDIA GPU**, scored against the **official COCO
val2017 annotations** with the reference `pycocotools` library — the same
pattern used in this author's other benchmark artifacts (from-scratch tracker
core, reference-library-scored). Until that run happens, `RESULTS.md` contains
no numbers, only the template.

See [`PLAN.md`](PLAN.md) for the full design, scope decisions, and step-by-step
build plan for whoever (human or agent) implements this next.
