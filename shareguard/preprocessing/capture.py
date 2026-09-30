"""Screen capture via `mss` (GDI-based on Windows; works without admin rights).

Windows Graphics Capture would be faster and zero-copy; it is on the roadmap.
"""
from __future__ import annotations

import numpy as np


class ScreenCapture:
    def __init__(self, monitor: int = 1, region: tuple[int, int, int, int] | None = None):
        import mss  # imported lazily so tests run headless
        self._sct = mss.mss()
        if region:
            left, top, width, height = region
            self._mon = {"left": left, "top": top, "width": width, "height": height}
        else:
            self._mon = self._sct.monitors[monitor]

    def grab(self) -> np.ndarray:
        img = np.asarray(self._sct.grab(self._mon))  # BGRA
        return np.ascontiguousarray(img[:, :, 2::-1])  # -> RGB
