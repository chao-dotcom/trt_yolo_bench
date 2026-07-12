#!/usr/bin/env python3
"""Score COCO-format detections against COCO val annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dets", type=Path, required=True, help="COCO-format detections JSON")
    parser.add_argument("--gt", type=Path, default=Path("data/instances_val2017.json"))
    parser.add_argument("--slice", type=Path, default=Path("data/val_slice.json"))
    return parser.parse_args()


def load_slice_ids(slice_path: Path) -> list[int]:
    records = json.loads(slice_path.read_text(encoding="utf-8"))
    ids = [int(record["id"]) for record in records]
    if not ids:
        raise ValueError(f"No image IDs found in slice: {slice_path}")
    return ids


def main() -> None:
    args = parse_args()

    if not args.dets.exists():
        raise FileNotFoundError(f"Detections file not found: {args.dets}")
    if not args.gt.exists():
        raise FileNotFoundError(f"Ground-truth annotations file not found: {args.gt}")
    if not args.slice.exists():
        raise FileNotFoundError(f"Slice file not found: {args.slice}")

    slice_img_ids = load_slice_ids(args.slice)

    coco_gt = COCO(str(args.gt))
    coco_dt = coco_gt.loadRes(str(args.dets))

    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.imgIds = slice_img_ids
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    print("\n[summary]")
    print(f"AP@[.5:.95]: {evaluator.stats[0]:.4f}")
    print(f"AP@.5:      {evaluator.stats[1]:.4f}")


if __name__ == "__main__":
    main()
