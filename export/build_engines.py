#!/usr/bin/env python3
"""Build TensorRT engines at FP32, FP16, and INT8."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorrt as trt

try:
    import pycuda.autoinit  # noqa: F401
except Exception as exc:  # pragma: no cover - environment dependent
    raise RuntimeError(
        "pycuda.autoinit import failed. Install pycuda in the CUDA environment before building engines."
    ) from exc

from calibrator import CocoInt8EntropyCalibrator, load_slice_records, resolve_image_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, default=Path("models/yolo11n.onnx"))
    parser.add_argument("--engines-dir", type=Path, default=Path("engines"))
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--calib-slice", type=Path, default=Path("data/calib_slice.json"))
    parser.add_argument("--calib-image-dir", type=Path, default=Path("data/train2017"))
    parser.add_argument("--calib-cache", type=Path, default=Path("calib.cache"))
    parser.add_argument("--calib-batch-size", type=int, default=8)
    parser.add_argument("--input-size", type=int, default=640)
    return parser.parse_args()


def build_serialized_engine(
    onnx_path: Path,
    precision: str,
    logger: trt.ILogger,
    workspace_bytes: int,
    calibrator: CocoInt8EntropyCalibrator | None,
) -> bytes:
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    if not parser.parse(onnx_path.read_bytes()):
        details = "\n".join(
            f"  [{i}] {parser.get_error(i)}" for i in range(parser.num_errors)
        )
        raise RuntimeError(f"ONNX parse failed:\n{details}")

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    else:
        config.max_workspace_size = workspace_bytes

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator is None:
            raise ValueError("INT8 requested but calibrator is None")
        config.int8_calibrator = calibrator

    if hasattr(builder, "build_serialized_network"):
        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError(f"TensorRT returned None while building {precision}")
        return bytes(plan)

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError(f"TensorRT returned None while building {precision}")
    return bytes(engine.serialize())


def main() -> None:
    args = parse_args()
    logger = trt.Logger(trt.Logger.INFO)

    if not args.onnx.exists():
        raise FileNotFoundError(f"ONNX file not found: {args.onnx}")

    args.engines_dir.mkdir(parents=True, exist_ok=True)
    workspace_bytes = int(args.workspace_gb * (1 << 30))

    calibrator = None
    if not args.calib_slice.exists():
        print(f"[warn] calibration slice missing: {args.calib_slice}")
    else:
        records = load_slice_records(args.calib_slice)
        image_paths = resolve_image_paths(records, args.calib_image_dir)
        calibrator = CocoInt8EntropyCalibrator(
            image_paths=image_paths,
            cache_path=args.calib_cache,
            input_hw=(args.input_size, args.input_size),
            batch_size=args.calib_batch_size,
        )

    targets = ("fp32", "fp16", "int8")
    for precision in targets:
        if precision == "int8" and calibrator is None:
            print("[skip] INT8 build skipped because calibration data is unavailable")
            continue

        print(f"[build] precision={precision}")
        plan = build_serialized_engine(
            onnx_path=args.onnx,
            precision=precision,
            logger=logger,
            workspace_bytes=workspace_bytes,
            calibrator=calibrator,
        )

        engine_path = args.engines_dir / f"yolo11n_{precision}.engine"
        engine_path.write_bytes(plan)
        print(f"[write] {engine_path} ({len(plan) / (1024 * 1024):.2f} MB)")

    print("[done] build step finished")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[error] {exc}")
        sys.exit(1)
