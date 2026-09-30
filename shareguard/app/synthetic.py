"""Deterministic synthetic screens with planted secrets and ground-truth boxes.

Used by demo mode and benchmark mode, so both work fully offline and the same
script produces the same frames on every machine. Fonts: DejaVu Sans Mono
(Bitstream Vera licence) or an OFL font such as Cascadia/JetBrains Mono.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..rules.detectors import _VERHOEFF_D, _VERHOEFF_P

W, H = 1280, 800
FONT_CANDIDATES = [
    os.environ.get("SHAREGUARD_FONT", ""),
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    r"C:\Windows\Fonts\CascadiaMono.ttf",
    r"C:\Windows\Fonts\CascadiaCode.ttf",
    "/Library/Fonts/JetBrainsMono-Regular.ttf",
]


def font(size: int) -> ImageFont.ImageFont:
    for p in FONT_CANDIDATES:
        if p and os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default(size)


_INV = [0, 4, 3, 2, 1, 5, 6, 7, 8, 9]


def make_aadhaar(rng: random.Random) -> str:
    body = str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(10))
    c = 0
    for i, ch in enumerate(reversed(body)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[(i + 1) % 8][int(ch)]]
    full = body + str(_INV[c])
    return f"{full[:4]} {full[4:8]} {full[8:]}"


def make_card(rng: random.Random) -> str:
    body = [4] + [rng.randint(0, 9) for _ in range(14)]
    total = 0
    for i, d in enumerate(reversed(body)):
        if i % 2 == 0:
            d *= 2
            d -= 9 if d > 9 else 0
        total += d
    digits = "".join(map(str, body)) + str((10 - total % 10) % 10)
    return " ".join(digits[i:i + 4] for i in range(0, 16, 4))


def make_pan(rng: random.Random) -> str:
    L = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(rng.choice(L) for _ in range(3)) + "P" + rng.choice(L) + \
        "".join(str(rng.randint(0, 9)) for _ in range(4)) + rng.choice(L)


def make_key(rng: random.Random) -> str:
    A = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "AKIA" + "".join(rng.choice(A) for _ in range(16))


@dataclass
class Screen:
    image: np.ndarray
    secrets: list[dict] = field(default_factory=list)   # {id,type,text,box}
    safe_boxes: list[tuple[int, int, int, int]] = field(default_factory=list)


class Canvas:
    def __init__(self, dark: bool, size: int):
        self.bg = (30, 30, 30) if dark else (250, 250, 250)
        self.fg = (220, 220, 220) if dark else (20, 20, 20)
        self.im = Image.new("RGB", (W, H), self.bg)
        self.d = ImageDraw.Draw(self.im)
        self.f = font(size)
        self.secrets: list[dict] = []
        self.safe: list[tuple[int, int, int, int]] = []

    def text(self, xy, s, secret_type=None, sid=None):
        box = self.d.textbbox(xy, s, font=self.f)
        self.d.text(xy, s, fill=self.fg, font=self.f)
        if secret_type:
            self.secrets.append({"id": sid, "type": secret_type, "text": s, "box": box})
        else:
            self.safe.append(box)
        return box

    def labelled(self, xy, label, value, secret_type, sid):
        # the label itself is context, not counted as safe text to reveal
        lb = self.d.textbbox(xy, label, font=self.f)
        self.d.text(xy, label, fill=self.fg, font=self.f)
        self.text((lb[2] + 12, xy[1]), value, secret_type, sid)

    def done(self) -> Screen:
        return Screen(np.asarray(self.im).copy(), self.secrets, self.safe)


def notes_screen(dark=False, size=22) -> Screen:
    c = Canvas(dark, size)
    c.text((40, 30), "Sprint notes - ShareGuard demo")
    body = ["Agenda: review latency charts and the unblur rule",
            "Action items: finish the benchmark harness",
            "Remember to test Teams and Google Meet sharing",
            "Deadline tonight, keep the scope small",
            "The quick brown fox jumps over the lazy dog"]
    for i, s in enumerate(body):
        c.text((40, 90 + i * 44), s)
    return c.done()


def form_screen(seed=7, dark=False, size=22) -> Screen:
    rng = random.Random(seed)
    c = Canvas(dark, size)
    c.text((40, 30), "Customer KYC form")
    c.labelled((40, 100), "Name:", "Priya Venkataraman", "name", "name1")
    c.labelled((40, 150), "Aadhaar:", make_aadhaar(rng), "aadhaar", "aad1")
    c.labelled((40, 200), "PAN:", make_pan(rng), "pan", "pan1")
    c.labelled((40, 250), "Card:", make_card(rng), "card", "card1")
    c.text((40, 330), "Please verify the documents before approval")
    return c.done()


def terminal_frames(seed=11, dark=True, size=20) -> list[Screen]:
    """A key being typed one character at a time into a .env file."""
    rng = random.Random(seed)
    key = make_key(rng)
    frames = []
    for n in range(0, len(key) + 1, 2):
        c = Canvas(dark, size)
        c.text((30, 30), "$ cat .env")
        c.text((30, 70), "DEBUG=false")
        c.text((30, 110), "REGION=ap-south-1")
        typed = key[:n]
        if typed:
            lb = c.d.textbbox((30, 150), "AWS_ACCESS_KEY_ID=", font=c.f)
            c.d.text((30, 150), "AWS_ACCESS_KEY_ID=", fill=c.fg, font=c.f)
            c.text((lb[2], 150), typed, "aws_key", "key1")
        else:
            c.text((30, 150), "AWS_ACCESS_KEY_ID=")
        frames.append(c.done())
    return frames


def demo_script(fps: float = 10.0) -> list[tuple[str, Screen]]:
    """(event, screen) per frame: idle notes -> window switch -> typing a key."""
    seq: list[tuple[str, Screen]] = []
    n = notes_screen()
    seq += [("idle" if i else "start", n) for i in range(int(fps * 3))]
    f = form_screen()
    seq += [("switch" if i == 0 else "idle", f) for i in range(int(fps * 4))]
    for j, s in enumerate(terminal_frames()):
        seq += [("switch" if j == 0 else "type", s)] + [("idle", s)] * int(fps * 0.3)
    seq += [("idle", seq[-1][1])] * int(fps * 3)
    return seq
