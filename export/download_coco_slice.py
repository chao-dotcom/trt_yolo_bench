#!/usr/bin/env python3
"""Create deterministic COCO slices and download only required images."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Iterable

import requests

ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
VAL_IMAGE_URL = "http://images.cocodataset.org/val2017/{file_name}"
TRAIN_IMAGE_URL = "http://images.cocodataset.org/train2017/{file_name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--calib-count", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def download_file(url: str, destination: Path, timeout_s: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[skip] {destination} already exists")
        return

    print(f"[download] {url} -> {destination}")
    with requests.get(url, stream=True, timeout=timeout_s) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)


def extract_annotations(zip_path: Path, data_dir: Path) -> tuple[Path, Path]:
    val_json = data_dir / "instances_val2017.json"
    train_json = data_dir / "instances_train2017.json"
    if val_json.exists() and train_json.exists():
        print("[skip] annotation JSON files already extracted")
        return val_json, train_json

    print(f"[extract] {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        for member, output_path in (
            ("annotations/instances_val2017.json", val_json),
            ("annotations/instances_train2017.json", train_json),
        ):
            if output_path.exists():
                print(f"[skip] {output_path} already exists")
                continue
            with archive.open(member) as source, output_path.open("wb") as target:
                target.write(source.read())
            print(f"[write] {output_path}")

    return val_json, train_json


def build_slice(instances_path: Path, count: int, output_path: Path) -> list[dict[str, object]]:
    payload = json.loads(instances_path.read_text(encoding="utf-8"))
    images = sorted(payload["images"], key=lambda img: img["id"])
    sliced = [{"id": img["id"], "file_name": img["file_name"]} for img in images[:count]]

    output_path.write_text(json.dumps(sliced, indent=2), encoding="utf-8")
    print(f"[write] {output_path} ({len(sliced)} entries)")
    return sliced


def download_images(
    records: Iterable[dict[str, object]],
    url_template: str,
    destination_dir: Path,
    timeout_s: int,
) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        file_name = str(record["file_name"])
        destination = destination_dir / file_name
        download_file(url_template.format(file_name=file_name), destination, timeout_s=timeout_s)


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    annotations_zip = data_dir / "annotations_trainval2017.zip"
    download_file(ANNOTATIONS_URL, annotations_zip, timeout_s=args.timeout)

    val_instances, train_instances = extract_annotations(annotations_zip, data_dir)
    val_slice_path = data_dir / "val_slice.json"
    calib_slice_path = data_dir / "calib_slice.json"

    val_slice = build_slice(val_instances, args.val_count, val_slice_path)
    calib_slice = build_slice(train_instances, args.calib_count, calib_slice_path)

    download_images(val_slice, VAL_IMAGE_URL, data_dir / "val2017", timeout_s=args.timeout)
    download_images(calib_slice, TRAIN_IMAGE_URL, data_dir / "train2017", timeout_s=args.timeout)

    print("[done] deterministic slices and required images are ready")


if __name__ == "__main__":
    main()
