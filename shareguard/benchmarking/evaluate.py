"""Run the pipeline over a scripted synthetic replay and score it.

Metrics (all measured, nothing assumed):
- per-secret miss: a secret is missed if ANY of its pixels' tiles was shown
  unblurred in ANY output frame (unit = secret, per the team's scoring rule)
- invariant violations: shown tiles not pixel-identical to a cleared scan (must be 0)
- time-to-readable after each screen event: seconds until >= `frac` of the
  safe-text tiles are shown
- OCR latency per scan (p50/p95)
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict

import numpy as np

from ..app.pipeline import Config, Pipeline
from ..app.synthetic import demo_script
from ..preprocessing.tiles import tile_hashes


def _tiles_for(box, tile):
    x0, y0, x1, y1 = box
    return {(r, c) for r in range(y0 // tile, y1 // tile + 1) for c in range(x0 // tile, x1 // tile + 1)}


def _ink_shown(img, box, tiles, tile) -> bool:
    """True if any shown tile contains actual glyph pixels of the secret
    (text bounding boxes include blank bearing space, which reveals nothing)."""
    x0, y0, x1, y1 = box
    bg = img[y0, max(0, x0 - 2)]
    for r, c in tiles:
        a = img[max(y0, r * tile):min(y1, (r + 1) * tile), max(x0, c * tile):min(x1, (c + 1) * tile)]
        if a.size and (a != bg).any():
            return True
    return False


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def run_replay(engine, cfg: Config | None = None, fps: float = 10.0, realtime: bool = True,
               frac: float = 0.9, save_frames: str | None = None) -> dict:
    cfg = cfg or Config()
    pipe = Pipeline(engine, cfg, synchronous=not realtime)
    script = demo_script(fps)
    missed: dict[str, dict] = {}
    seen: set[str] = set()
    violations = 0
    ocr_ms: list[float] = []
    events: list[dict] = []
    open_event = None
    t_start = time.perf_counter()
    frames_out = []
    for i, (ev, scr) in enumerate(script):
        target = t_start + i / fps
        if realtime:
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        now = time.perf_counter() if realtime else target
        out, st = pipe.process(scr.image, now)
        violations += st.violations
        if st.ocr_ms is not None:
            ocr_ms.append(st.ocr_ms)
        scan = pipe.last_scan()
        shown = set()
        # recompute the shown set exactly as the compositor did
        if scan is not None:
            cur = tile_hashes(scr.image, cfg.tile)
            if cur.shape == scan.hashes.shape:
                shown = set(zip(*np.nonzero((scan.hashes == cur) & ~scan.blocked)))
        for s in scr.secrets:
            seen.add(s["id"])
            if s["id"] not in missed and _ink_shown(scr.image, s["box"], _tiles_for(s["box"], cfg.tile) & shown, cfg.tile):
                missed[s["id"]] = {"type": s["type"], "text": s["text"], "frame": i}
        safe_tiles = set().union(*[_tiles_for(b, cfg.tile) for b in scr.safe_boxes]) if scr.safe_boxes else set()
        secret_tiles = set().union(*[_tiles_for(s["box"], cfg.tile) for s in scr.secrets]) if scr.secrets else set()
        safe_tiles -= secret_tiles
        readable = len(safe_tiles & shown) / len(safe_tiles) if safe_tiles else 1.0
        if ev in ("start", "switch"):
            open_event = {"event": ev, "frame": i, "t0": now}
            events.append(open_event)
        if open_event is not None and "ttr_s" not in open_event and readable >= frac:
            open_event["ttr_s"] = round(now - open_event["t0"], 3)
        if save_frames and i % max(1, int(fps // 2)) == 0:
            frames_out.append(out)
    pipe.close()
    ttr = [e["ttr_s"] for e in events if "ttr_s" in e]
    if not realtime:  # synchronous replay: OCR takes zero simulated time, so TTR is meaningless
        events, ttr = [{"event": e["event"], "frame": e["frame"]} for e in events], []
    res = {
        "engine": engine.name, "device": engine.device, "realtime": realtime, "fps": fps,
        "config": asdict(cfg), "frames": len(script),
        "secrets_planted": len(seen), "secrets_missed": len(missed), "missed_detail": missed,
        "invariant_violations": violations,
        "ocr_ms_p50": pct(ocr_ms, 50), "ocr_ms_p95": pct(ocr_ms, 95), "scans": len(ocr_ms),
        "time_to_readable_s": events, "ttr_p50": pct(ttr, 50), "ttr_p95": pct(ttr, 95),
    }
    if save_frames and frames_out:
        from PIL import Image
        ims = [Image.fromarray(f).resize((640, 400)) for f in frames_out]
        ims[0].save(save_frames, save_all=True, append_images=ims[1:], duration=int(1000 / 2), loop=0)
    return res


if __name__ == "__main__":
    from ..inference.base import get_engine
    print(json.dumps(run_replay(get_engine("rapidocr")), indent=2, default=str))
