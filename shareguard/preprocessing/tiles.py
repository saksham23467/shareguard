"""Tile hashing: change detection and the pixel-identity check before unblur."""
from __future__ import annotations

import hashlib

import numpy as np


def tile_hashes(frame: np.ndarray, tile: int) -> np.ndarray:
    """Return an (rows, cols) object array of 128-bit blake2b digests per tile."""
    h, w = frame.shape[:2]
    rows, cols = (h + tile - 1) // tile, (w + tile - 1) // tile
    out = np.empty((rows, cols), dtype=object)
    for r in range(rows):
        for c in range(cols):
            blk = np.ascontiguousarray(frame[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile])
            out[r, c] = hashlib.blake2b(blk.tobytes(), digest_size=16).digest()
    return out


def changed_mask(prev: np.ndarray | None, cur: np.ndarray) -> np.ndarray:
    if prev is None or prev.shape != cur.shape:
        return np.ones(cur.shape, dtype=bool)
    return prev != cur
