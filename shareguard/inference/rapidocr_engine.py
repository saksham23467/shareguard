"""RapidOCR (Apache-2.0; PP-OCR detection + recognition in ONNX) on ONNX Runtime CPU.

This is the engine the app uses tonight. It runs on the CPU: its models use
dynamic input shapes, which the QNN NPU backend does not accept, so we do NOT
claim NPU execution for it. NPU numbers come from the fixed-shape Qualcomm AI Hub
EasyOCR models, measured separately (see scripts/aihub_profile.py and
`main npu-probe`).
"""
from __future__ import annotations

import numpy as np

from .base import Line, OCREngine


class RapidOCREngine(OCREngine):
    name = "rapidocr"
    device = "cpu"

    def __init__(self):
        from rapidocr_onnxruntime import RapidOCR
        self._eng = RapidOCR()

    def read(self, frame_rgb: np.ndarray) -> list[Line]:
        result, _ = self._eng(frame_rgb[:, :, ::-1].copy())  # engine expects BGR
        lines: list[Line] = []
        for pts, text, score in result or []:
            xs = [int(p[0]) for p in pts]
            ys = [int(p[1]) for p in pts]
            lines.append(Line((min(xs), min(ys), max(xs), max(ys)), str(text), float(score)))
        return lines
