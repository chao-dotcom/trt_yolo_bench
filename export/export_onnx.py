#!/usr/bin/env python3
"""Export YOLO model to ONNX and validate output tensor shape."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from ultralytics import YOLO


EXPECTED_SHAPE = [1, 84, 8400]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=str, default="yolo11n.pt")
    parser.add_argument("--out-dir", type=Path, default=Path("models"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def read_shape(onnx_path: Path) -> list[int | str]:
    model = onnx.load(str(onnx_path))
    output = model.graph.output[0]
    dims: list[int | str] = []
    for dim in output.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=False,
    )

    exported_path = Path(exported)
    if not exported_path.exists():
        raise FileNotFoundError(f"Export did not produce ONNX file: {exported_path}")

    final_path = args.out_dir / exported_path.name
    if final_path.resolve() != exported_path.resolve():
        final_path.write_bytes(exported_path.read_bytes())
        print(f"[write] copied ONNX to {final_path}")
    else:
        print(f"[info] ONNX already in desired location: {final_path}")

    shape = read_shape(final_path)
    print(f"[info] ONNX output shape: {shape}")
    if shape != EXPECTED_SHAPE:
        raise ValueError(
            "Unexpected ONNX output shape. Update C++ decode logic to match this shape before benchmarking. "
            f"Expected {EXPECTED_SHAPE}, got {shape}."
        )

    print("[done] ONNX export complete and shape validated")


if __name__ == "__main__":
    main()
