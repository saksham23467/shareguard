"""Share window: shows the redacted screen. Share THIS window in Teams/Meet.

Tkinter is used because it ships with Python on Windows ARM64, so the app has no
extra GUI dependency. Press H to toggle the HUD (red = blurred tiles), Q to quit.
Untested on Windows as of submission night; see docs/TESTING.md.
"""
from __future__ import annotations

import time
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageTk


class Viewer:
    def __init__(self, title="ShareGuard - share this window", scale=0.6):
        self.root = tk.Tk()
        self.root.title(title)
        self.scale, self.hud, self.alive = scale, True, True
        self.label = tk.Label(self.root)
        self.label.pack()
        self.root.bind("<h>", lambda e: setattr(self, "hud", not self.hud))
        self.root.bind("<q>", lambda e: self.close())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._fps_t, self._n = time.perf_counter(), 0

    def close(self):
        self.alive = False
        self.root.destroy()

    def show(self, frame: np.ndarray, shown_mask: np.ndarray | None, tile: int, text: str):
        im = Image.fromarray(frame)
        if self.hud:
            d = ImageDraw.Draw(im, "RGBA")
            if shown_mask is not None:
                for r, c in zip(*np.nonzero(~shown_mask)):
                    d.rectangle([c * tile, r * tile, (c + 1) * tile - 1, (r + 1) * tile - 1],
                                outline=(255, 60, 60, 90))
            d.rectangle([0, 0, 520, 28], fill=(0, 0, 0, 170))
            d.text((8, 7), text, fill=(255, 255, 255, 255))
        w, h = im.size
        im = im.resize((int(w * self.scale), int(h * self.scale)))
        self._tk = ImageTk.PhotoImage(im)
        self.label.configure(image=self._tk)
        self.root.update_idletasks()
        self.root.update()
