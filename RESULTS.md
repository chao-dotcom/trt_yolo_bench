# RESULTS — measured, not fabricated

**Status: PENDING.** Nothing in this file has been run yet. Every cell below
must come from actually executing the code in `export/`, `harness/`, and
`eval/` on a real NVIDIA GPU (see `PLAN.md` §0 — the authoring machine has no
GPU, so this file cannot be filled in until the code runs somewhere that has
one). Do not replace `PENDING` with an estimated, typical, or "expected"
number — that is a fabricated result even if it turns out to be close.

See `PLAN.md` §7 for the exact rules this file follows once real numbers
exist.

---

## Environment

*(fill in exactly, once run — GPU model, driver version, CUDA version,
TensorRT version, OS. TensorRT numbers are not portable across these.)*

- GPU: PENDING
- Driver: PENDING
- CUDA: PENDING
- TensorRT: PENDING
- OS: PENDING

## Headline table

| Precision | AP@[.5:.95] | AP@.5 | Mean latency (ms) | p99 (ms) | Throughput (FPS) | Engine size (MB) |
|---|---|---|---|---|---|---|
| FP32 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| FP16 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |
| INT8 | PENDING | PENDING | PENDING | PENDING | PENDING | PENDING |

## Exact commands used

```
PENDING — paste the exact export_onnx.py / build_engines.py / harness / score_coco.py
invocations here once run, so the table above is reproducible from this file alone.
```

## Verbatim `COCOeval.summarize()` output — per precision

```
PENDING
```

## Verbatim benchmark harness output — per precision

```
PENDING
```

## How to read these numbers (fill in once real)

*(e.g. is the INT8 AP drop in line with published YOLO INT8 quantization
literature? Is latency dominated by preprocessing, inference, or
postprocessing? Any surprises, and the honest explanation for them — see
`PLAN.md` §7's note on explaining regressions rather than hiding them.)*

## Stretch: RT-DETR (optional, only if attempted)

*(only add this section if the RT-DETR comparison in `PLAN.md` §2/§9 was
actually run — otherwise leave it out entirely rather than a PENDING
placeholder, since it's explicitly optional scope.)*
