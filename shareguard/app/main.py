"""ShareGuard command line.

  python -m shareguard.app.main check                 # hardware + runtime report
  python -m shareguard.app.main demo [--gif out.gif]  # offline scripted replay, scored
  python -m shareguard.app.main live [--region x y w h] [--offline]
  python -m shareguard.app.main bench                 # per-stage latency -> results/*.json
  python -m shareguard.app.main npu-probe MODEL.onnx  # one model on npu, gpu, cpu
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time

log = logging.getLogger("shareguard")


def setup_logging(level: str):
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(level=getattr(logging, level.upper()),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler("logs/shareguard.log")])


def enforce_offline():
    """Offline mode: block all outbound sockets in this process."""
    def _deny(*a, **k):
        raise OSError("ShareGuard offline mode: network disabled")
    socket.socket.connect = _deny  # type: ignore[assignment]
    socket.create_connection = _deny  # type: ignore[assignment]
    log.info("offline mode on: outbound network blocked in-process")


def cmd_check(_):
    from ..hardware.detect import detect
    info = detect()
    print(json.dumps({**info.__dict__, "windows_arm64": info.is_windows_arm64,
                      "qnn_available": info.has_qnn}, indent=2))
    if not info.has_qnn:
        print("QNNExecutionProvider not found: NPU/GPU paths disabled, CPU fallback will be used.")


def cmd_demo(a):
    from ..app.pipeline import Config
    from ..benchmarking.evaluate import run_replay
    from ..inference.base import get_engine
    res = run_replay(get_engine(a.engine), Config(conf_threshold=a.conf), fps=a.fps,
                     realtime=not a.fast, save_frames=a.gif)
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=2, default=str))
    if a.save:
        from ..benchmarking.bench import save
        print("saved", save({"demo": res}))


def cmd_bench(a):
    from ..benchmarking.bench import pipeline_bench, save
    res = pipeline_bench(a.engine, a.n)
    print(json.dumps(res, indent=2, default=str))
    print("saved", save({"pipeline_bench": res}))


def cmd_probe(a):
    from ..benchmarking.bench import model_probe, save
    out = []
    for dev in a.devices:
        try:
            out.append(model_probe(a.model, dev, a.n))
        except Exception as e:  # noqa: BLE001
            out.append({"requested": dev, "error": str(e)})
        print(json.dumps(out[-1], indent=2))
    print("saved", save({"model_probe": out}))


def cmd_live(a):
    import numpy as np
    from ..app.pipeline import Config, Pipeline
    from ..frontend.viewer import Viewer
    from ..inference.base import get_engine
    from ..preprocessing.capture import ScreenCapture
    from ..preprocessing.tiles import tile_hashes
    cfg = Config(conf_threshold=a.conf)
    eng = get_engine(a.engine)
    cap = ScreenCapture(a.monitor, tuple(a.region) if a.region else None)
    pipe, view = Pipeline(eng, cfg), Viewer()
    last_ocr = None
    try:
        while view.alive:
            t0 = time.perf_counter()
            frame = cap.grab()
            out, st = pipe.process(frame)
            if st.ocr_ms is not None:
                last_ocr = st.ocr_ms
            scan = pipe.last_scan()
            mask = None
            if scan is not None:
                cur = tile_hashes(frame, cfg.tile)
                if cur.shape == scan.hashes.shape:
                    mask = (scan.hashes == cur) & ~scan.blocked
            fps = 1 / max(1e-6, time.perf_counter() - t0)
            view.show(out, mask, cfg.tile,
                      f"{eng.name}/{eng.device}  shown {st.shown_tiles}/{st.total_tiles}  "
                      f"ocr {last_ocr or 0:.0f} ms  {fps:.1f} fps  violations {st.violations}")
    finally:
        pipe.close()


def main(argv=None):
    p = argparse.ArgumentParser(prog="shareguard")
    p.add_argument("--log", default="info")
    p.add_argument("--offline", action="store_true", help="block all network access")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    d = sub.add_parser("demo")
    d.add_argument("--engine", default="rapidocr")
    d.add_argument("--fps", type=float, default=10)
    d.add_argument("--conf", type=float, default=0.80)
    d.add_argument("--fast", action="store_true", help="synchronous, not wall-clock")
    d.add_argument("--gif")
    d.add_argument("--save", action="store_true")
    d.set_defaults(fn=cmd_demo)
    b = sub.add_parser("bench")
    b.add_argument("--engine", default="rapidocr")
    b.add_argument("-n", type=int, default=20)
    b.set_defaults(fn=cmd_bench)
    q = sub.add_parser("npu-probe")
    q.add_argument("model")
    q.add_argument("--devices", nargs="+", default=["npu", "gpu", "cpu"])
    q.add_argument("-n", type=int, default=100)
    q.set_defaults(fn=cmd_probe)
    lv = sub.add_parser("live")
    lv.add_argument("--engine", default="rapidocr")
    lv.add_argument("--monitor", type=int, default=1)
    lv.add_argument("--region", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    lv.add_argument("--conf", type=float, default=0.80)
    lv.set_defaults(fn=cmd_live)
    a = p.parse_args(argv)
    setup_logging(a.log)
    if a.offline:
        enforce_offline()
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
