"""INT8 entropy calibrator for TensorRT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt


def letterbox_bgr(
    image: np.ndarray,
    target_size: tuple[int, int],
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, int, int]:
    target_h, target_w = target_size
    src_h, src_w = image.shape[:2]

    scale = min(target_w / src_w, target_h / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))

    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)

    pad_x = (target_w - resized_w) // 2
    pad_y = (target_h - resized_h) // 2
    canvas[pad_y : pad_y + resized_h, pad_x : pad_x + resized_w] = resized

    return canvas, scale, pad_x, pad_y


def preprocess_to_chw_float(image_bgr: np.ndarray, input_hw: tuple[int, int]) -> np.ndarray:
    boxed, _, _, _ = letterbox_bgr(image_bgr, target_size=input_hw)
    rgb = cv2.cvtColor(boxed, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    return np.ascontiguousarray(tensor)


def load_slice_records(slice_path: Path) -> list[dict[str, object]]:
    records = json.loads(slice_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Slice file must contain a list: {slice_path}")
    return records


def resolve_image_paths(records: Iterable[dict[str, object]], image_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for record in records:
        file_name = record.get("file_name")
        if not isinstance(file_name, str):
            raise ValueError(f"Record missing file_name: {record}")
        image_path = image_dir / file_name
        if not image_path.exists():
            raise FileNotFoundError(f"Calibration image not found: {image_path}")
        paths.append(image_path)
    return paths


class CocoInt8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        image_paths: Sequence[Path],
        cache_path: Path,
        input_hw: tuple[int, int] = (640, 640),
        batch_size: int = 8,
    ) -> None:
        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not image_paths:
            raise ValueError("Calibration requires at least one image")

        self.image_paths = list(image_paths)
        self.cache_path = cache_path
        self.input_hw = input_hw
        self.batch_size = batch_size

        self.current_index = 0
        channels = 3
        elems_per_image = channels * input_hw[0] * input_hw[1]
        self.device_input = cuda.mem_alloc(batch_size * elems_per_image * np.float32().nbytes)

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list[str]) -> list[int] | None:
        del names
        if self.current_index >= len(self.image_paths):
            return None

        end_index = min(self.current_index + self.batch_size, len(self.image_paths))
        batch_paths = self.image_paths[self.current_index : end_index]
        self.current_index = end_index

        batch = np.zeros((self.batch_size, 3, self.input_hw[0], self.input_hw[1]), dtype=np.float32)
        for i, image_path in enumerate(batch_paths):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read calibration image: {image_path}")
            batch[i] = preprocess_to_chw_float(image, self.input_hw)

        cuda.memcpy_htod(self.device_input, batch)
        return [int(self.device_input)]

    def read_calibration_cache(self) -> bytes | None:
        if self.cache_path.exists():
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_bytes(cache)
