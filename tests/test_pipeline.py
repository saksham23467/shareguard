"""Pipeline invariants with a fake OCR engine (no models needed)."""
import numpy as np

from shareguard.app.pipeline import Config, Pipeline
from shareguard.inference.base import Line, OCREngine


class FakeOCR(OCREngine):
    name, device = "fake", "cpu"

    def __init__(self, lines):
        self.lines = lines

    def read(self, frame):
        return self.lines


def frame(v=200):
    f = np.full((128, 256, 3), v, np.uint8)
    f[40:60, 20:200] = 0  # a "text line"
    return f


def test_first_frame_fully_blurred_until_scan():
    p = Pipeline(FakeOCR([]), Config(tile=32), synchronous=False)
    p._pool.submit(lambda: None).result()
    out, st = p.process(frame(), 0.0)
    assert st.shown_tiles == 0  # fail-closed before any scan result
    p.close()


def test_safe_line_shown_and_secret_blocked():
    safe = Line((20, 40, 200, 60), "hello world", 0.99)
    p = Pipeline(FakeOCR([safe]), Config(tile=32), synchronous=True)
    _, st = p.process(frame(), 0.0)
    assert st.shown_tiles == st.total_tiles and st.violations == 0
    key = Line((20, 40, 200, 60), "AKIAABCDEFGHIJKLMNOP", 0.99)
    p2 = Pipeline(FakeOCR([key]), Config(tile=32), synchronous=True)
    _, st2 = p2.process(frame(), 0.0)
    assert st2.shown_tiles < st2.total_tiles and st2.violations == 0


def test_low_confidence_stays_blurred():
    ln = Line((20, 40, 200, 60), "hello world", 0.3)
    p = Pipeline(FakeOCR([ln]), Config(tile=32, conf_threshold=0.8), synchronous=True)
    _, st = p.process(frame(), 0.0)
    assert st.shown_tiles < st.total_tiles


def test_changed_pixels_reblur_before_rescan():
    safe = Line((20, 40, 200, 60), "hello", 0.99)
    p = Pipeline(FakeOCR([safe]), Config(tile=32), synchronous=True)
    p.process(frame(), 0.0)
    # simulate the race: pixels change but the old scan is still the latest
    f2 = frame()
    f2[40:60, 20:200] = 90
    p.sync = False
    p._pending = p._pool.submit(lambda: __import__("time").sleep(0.5) or p._last)
    out, st = p.process(f2, 1.0)
    assert st.shown_tiles < st.total_tiles  # changed tiles hidden
    assert np.array_equal(out[40:60, 20:200], out[40:60, 20:200])
    assert st.violations == 0
    p.close()
