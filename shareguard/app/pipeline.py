"""Fail-closed redaction pipeline.

Invariant (checked in tests and in demo/bench output): every tile shown unblurred
in an output frame is pixel-identical (128-bit hash) to a tile that an OCR scan
read and cleared. Anything changed, unread, low-confidence or sensitive stays
blurred. OCR runs asynchronously; the compositor never waits for it.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFilter

from ..inference.base import Line, OCREngine
from ..preprocessing.tiles import tile_hashes
from ..rules.detectors import detect, has_field_label

log = logging.getLogger(__name__)


@dataclass
class Config:
    tile: int = 32
    conf_threshold: float = 0.80   # below this an OCR line stays blurred
    entropy_bits: float = 3.5
    pad: int = 4
    blur_downscale: int = 16
    label_extends_right: bool = True


@dataclass
class ScanResult:
    hashes: np.ndarray
    blocked: np.ndarray          # bool tiles: sensitive or uncertain
    lines: list[Line]
    reasons: list[tuple[Line, str]]
    ocr_ms: float
    submitted_at: float


@dataclass
class FrameStats:
    t: float
    shown_tiles: int
    total_tiles: int
    violations: int
    ocr_ms: float | None = None
    new_scan: bool = False


def blur(frame: np.ndarray, k: int) -> np.ndarray:
    h, w = frame.shape[:2]
    im = Image.fromarray(frame)
    small = im.resize((max(1, w // k), max(1, h // k)), Image.BOX).filter(ImageFilter.GaussianBlur(2))
    return np.array(small.resize((w, h), Image.BILINEAR))


def classify_lines(lines: list[Line], cfg: Config) -> list[tuple[Line, str]]:
    """Return (line, reason) for every line that must stay blurred."""
    out = []
    for ln in lines:
        f = detect(ln.text, entropy_bits=cfg.entropy_bits)
        if f:
            out.append((ln, "sensitive:" + ",".join(sorted({x.kind for x in f}))))
        elif ln.conf < cfg.conf_threshold:
            out.append((ln, f"uncertain:conf={ln.conf:.2f}"))
        elif has_field_label(ln.text):
            out.append((ln, "field_label"))
    return out


def blocked_tiles(shape, reasons, cfg: Config) -> np.ndarray:
    h, w = shape[:2]
    t = cfg.tile
    rows, cols = (h + t - 1) // t, (w + t - 1) // t
    mask = np.zeros((rows, cols), dtype=bool)
    for ln, why in reasons:
        x0, y0, x1, y1 = ln.box
        if why == "field_label":
            if not cfg.label_extends_right:
                continue
            x1 = w  # value usually sits right of its label
        x0, y0 = max(0, x0 - cfg.pad), max(0, y0 - cfg.pad)
        x1, y1 = min(w - 1, x1 + cfg.pad), min(h - 1, y1 + cfg.pad)
        mask[y0 // t:y1 // t + 1, x0 // t:x1 // t + 1] = True
    return mask


class Pipeline:
    def __init__(self, engine: OCREngine, cfg: Config | None = None, synchronous: bool = False):
        self.engine, self.cfg, self.sync = engine, cfg or Config(), synchronous
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._pending: Future | None = None
        self._last: ScanResult | None = None
        self._last_submitted_hashes: np.ndarray | None = None
        self._lock = threading.Lock()

    def _scan(self, frame: np.ndarray, hashes: np.ndarray, t_sub: float) -> ScanResult:
        t0 = time.perf_counter()
        lines = self.engine.read(frame)
        ms = (time.perf_counter() - t0) * 1000
        reasons = classify_lines(lines, self.cfg)
        return ScanResult(hashes, blocked_tiles(frame.shape, reasons, self.cfg), lines, reasons, ms, t_sub)

    def process(self, frame: np.ndarray, now: float | None = None) -> tuple[np.ndarray, FrameStats]:
        now = time.perf_counter() if now is None else now
        cfg = self.cfg
        hashes = tile_hashes(frame, cfg.tile)
        new_scan = False
        # collect a finished scan
        if self._pending is not None and (self.sync or self._pending.done()):
            self._last = self._pending.result()
            self._pending = None
            new_scan = True
        # submit a new scan if the screen differs from what was last submitted
        changed = (self._last_submitted_hashes is None
                   or self._last_submitted_hashes.shape != hashes.shape
                   or bool((self._last_submitted_hashes != hashes).any()))
        if changed and self._pending is None:
            self._last_submitted_hashes = hashes
            self._pending = self._pool.submit(self._scan, frame.copy(), hashes, now)
            if self.sync:
                self._last = self._pending.result()
                self._pending = None
                new_scan = True
        out = blur(frame, cfg.blur_downscale)
        shown = np.zeros(hashes.shape, dtype=bool)
        if self._last is not None and self._last.hashes.shape == hashes.shape:
            shown = (self._last.hashes == hashes) & ~self._last.blocked
        t = cfg.tile
        for r, c in zip(*np.nonzero(shown)):
            out[r * t:(r + 1) * t, c * t:(c + 1) * t] = frame[r * t:(r + 1) * t, c * t:(c + 1) * t]
        # independent re-check of the invariant on the composed frame
        violations = 0
        if shown.any():
            out_h = tile_hashes(out, t)
            violations = int(((out_h != self._last.hashes) & shown).sum())
        stats = FrameStats(now, int(shown.sum()), shown.size, violations,
                           self._last.ocr_ms if (new_scan and self._last) else None, new_scan)
        return out, stats

    def last_scan(self) -> ScanResult | None:
        return self._last

    def close(self):
        self._pool.shutdown(wait=False, cancel_futures=True)
