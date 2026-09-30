"""Benchmark mode: per-stage latency on the deterministic synthetic screens,
plus a raw per-model probe for any ONNX model on NPU / GPU / CPU.

Results are written to results/<timestamp>_<host>.json together with the
hardware report, so every number carries the machine it was measured on.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import time
from datetime import datetime

import numpy as np

from ..app.pipeline import Config, blur, classify_lines
from ..app.synthetic import form_screen, notes_screen, terminal_frames
from ..hardware.detect import detect, make_session
from ..preprocessing.tiles import tile_hashes


def _stats(ms: list[float]) -> dict:
    a = np.array(ms)
    return {"n": len(ms), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "mean_ms": float(a.mean()), "throughput_per_s": float(1000 / a.mean()) if a.mean() else None}


def _time(fn, n, warmup=3):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out


def _rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:  # noqa: BLE001 - psutil optional
        return None


def pipeline_bench(engine_name: str = "rapidocr", n: int = 20) -> dict:
    from ..inference.base import get_engine
    t0 = time.perf_counter()
    eng = get_engine(engine_name)
    load_ms = (time.perf_counter() - t0) * 1000
    cfg = Config()
    screens = {"notes": notes_screen().image, "form": form_screen().image,
               "terminal": terminal_frames()[-1].image}
    res = {"engine": engine_name, "device": eng.device, "model_load_ms": load_ms, "stages": {}}
    for name, img in screens.items():
        lines = eng.read(img)
        res["stages"][name] = {
            "tile_hash": _stats(_time(lambda: tile_hashes(img, cfg.tile), n)),
            "ocr": _stats(_time(lambda: eng.read(img), n, warmup=2)),
            "rules": _stats(_time(lambda: classify_lines(lines, cfg), n)),
            "blur": _stats(_time(lambda: blur(img, cfg.blur_downscale), n)),
            "lines_read": len(lines),
        }
    res["rss_mb"] = _rss_mb()
    return res


def model_probe(model_path: str, device: str, n: int = 100) -> dict:
    """Time one ONNX model (e.g. an AI Hub EasyOCR export) on a given device."""
    t0 = time.perf_counter()
    sess, actual = make_session(model_path, device)
    load_ms = (time.perf_counter() - t0) * 1000
    feeds = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        dtype = {"tensor(float)": np.float32, "tensor(uint8)": np.uint8,
                 "tensor(uint16)": np.uint16, "tensor(int64)": np.int64}.get(inp.type, np.float32)
        feeds[inp.name] = np.random.default_rng(0).random(shape).astype(dtype)
    ms = _time(lambda: sess.run(None, feeds), n, warmup=10)
    return {"model": os.path.basename(model_path), "requested": device, "actual_device": actual,
            "session_create_ms": load_ms, **_stats(ms)}


def save(result: dict, outdir: str = "results") -> str:
    os.makedirs(outdir, exist_ok=True)
    result = {"measured_at": datetime.now().isoformat(timespec="seconds"),
              "host": platform.node(), "hardware": detect().__dict__, **result}
    path = os.path.join(outdir, f"{datetime.now():%Y%m%d_%H%M%S}_{platform.node()}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return path
