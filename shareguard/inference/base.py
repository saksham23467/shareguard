"""Model abstraction layer. Every OCR engine returns the same Line records."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Line:
    box: tuple[int, int, int, int]  # x0, y0, x1, y1 in frame pixels
    text: str
    conf: float  # engine confidence in [0, 1]


class OCREngine(ABC):
    name: str = "base"
    device: str = "cpu"  # the device the engine actually ran on, never assumed

    @abstractmethod
    def read(self, frame_rgb: np.ndarray) -> list[Line]:
        ...


def get_engine(name: str) -> OCREngine:
    if name == "rapidocr":
        from .rapidocr_engine import RapidOCREngine
        return RapidOCREngine()
    raise ValueError(f"unknown OCR engine {name!r}")
