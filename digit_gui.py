#!/usr/bin/env python3
"""GUI: draw digits 0–9 with the mouse; the neural net predicts the class."""

from __future__ import annotations

import os
import sys
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox

import numpy as np

from nn import Sequential, Dense, ReLU
from nn.persist import load_weights

DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(DIR, "mnist_model.npz")

GRID = 28
CELL = 12
CANVAS_PX = GRID * CELL
PREDICT_DEBOUNCE_MS = 120


def build_model() -> Sequential:
    return Sequential(
        [
            Dense(784, 128, seed=None),
            ReLU(),
            Dense(128, 64, seed=None),
            ReLU(),
            Dense(64, 10, seed=None),
        ]
    )


def load_trained_model() -> Sequential:
    model = build_model()
    load_weights(model, WEIGHTS_PATH)
    return model


class DigitApp:
    def __init__(self) -> None:
        self.model = load_trained_model()
        self.grid = np.zeros((GRID, GRID), dtype=np.float64)
        self._predict_job: str | None = None

        self.root = tk.Tk()
        self.root.title("Digit recognition — draw 0 to 9")
        self.root.configure(bg="#F1F5F9")
        ui = tkfont.nametofont("TkDefaultFont").actual()["family"]
        title_f = tkfont.Font(family=ui, size=15, weight="bold")
        body_f = tkfont.Font(family=ui, size=11)
        small_f = tkfont.Font(family=ui, size=9)
        pred_f = tkfont.Font(family=ui, size=28, weight="bold")

        tk.Label(
            self.root,
            text="Draw a digit from 0 to 9",
            font=title_f,
            fg="#0F172A",
            bg="#F1F5F9",
        ).pack(pady=(18, 4))
        tk.Label(
            self.root,
            text="Use the mouse on the white area. The network predicts as you draw.",
            font=small_f,
            fg="#64748B",
            bg="#F1F5F9",
        ).pack()

        self.cv = tk.Canvas(
            self.root,
            width=CANVAS_PX,
            height=CANVAS_PX,
            bg="white",
            highlightthickness=2,
            highlightbackground="#CBD5E1",
        )
        self.cv.pack(pady=18)
        self.cv.bind("<B1-Motion>", self._paint)
        self.cv.bind("<ButtonPress-1>", self._paint)
        self.cv.bind("<ButtonRelease-1>", self._on_release)

        row = tk.Frame(self.root, bg="#F1F5F9")
        row.pack(pady=8)
        tk.Button(
            row,
            text="Clear",
            command=self._clear,
            font=body_f,
            padx=18,
            pady=6,
        ).pack(side=tk.LEFT, padx=6)
        tk.Button(
            row,
            text="Predict",
            command=self._predict,
            font=body_f,
            padx=18,
            pady=6,
        ).pack(side=tk.LEFT, padx=6)

        self.pred_var = tk.StringVar(value="—")
        tk.Label(
            self.root,
            text="Predicted digit",
            font=small_f,
            fg="#64748B",
            bg="#F1F5F9",
        ).pack(pady=(4, 0))
        tk.Label(
            self.root,
            textvariable=self.pred_var,
            font=pred_f,
            fg="#2563EB",
            bg="#F1F5F9",
        ).pack(pady=(0, 4))

        self.detail_var = tk.StringVar(value="")
        tk.Label(
            self.root,
            textvariable=self.detail_var,
            font=small_f,
            fg="#64748B",
            bg="#F1F5F9",
            justify=tk.CENTER,
        ).pack(pady=(0, 20))

    def _cell(self, event: tk.Event) -> tuple[int, int]:
        c = int(np.clip(event.x // CELL, 0, GRID - 1))
        r = int(np.clip(event.y // CELL, 0, GRID - 1))
        return r, c

    def _paint(self, event: tk.Event) -> None:
        r, c = self._cell(event)
        # Thin pen: strong center, light N/E/S/W so lines stay connected without a fat blob
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < GRID and 0 <= cc < GRID:
                add = 0.4 if (dr, dc) == (0, 0) else 0.12
                self.grid[rr, cc] = min(1.0, self.grid[rr, cc] + add)
        self._redraw_canvas()
        self._schedule_predict()

    def _schedule_predict(self) -> None:
        if self._predict_job is not None:
            self.root.after_cancel(self._predict_job)
        self._predict_job = self.root.after(PREDICT_DEBOUNCE_MS, self._predict)

    def _on_release(self, event: tk.Event) -> None:
        if self._predict_job is not None:
            self.root.after_cancel(self._predict_job)
            self._predict_job = None
        self._predict()

    def _redraw_canvas(self) -> None:
        self.cv.delete("all")
        for r in range(GRID):
            for c in range(GRID):
                v = self.grid[r, c]
                if v <= 0:
                    continue
                gray = int(255 * (1.0 - v))
                color = f"#{gray:02x}{gray:02x}{gray:02x}"
                x0, y0 = c * CELL, r * CELL
                self.cv.create_rectangle(
                    x0, y0, x0 + CELL, y0 + CELL,
                    fill=color, outline="",
                )

    def _clear(self) -> None:
        if self._predict_job is not None:
            self.root.after_cancel(self._predict_job)
            self._predict_job = None
        self.grid.fill(0.0)
        self.cv.delete("all")
        self.pred_var.set("—")
        self.detail_var.set("")

    def _predict(self, event: tk.Event | None = None) -> None:
        self._predict_job = None
        x = self.grid.reshape(1, 784)
        if x.max() < 1e-6:
            self.pred_var.set("—")
            self.detail_var.set("Draw a digit first.")
            return
        logits = self.model.forward(x)
        probs = self._softmax(logits[0])
        digit = int(np.argmax(probs))
        self.pred_var.set(str(digit))
        top = np.argsort(probs)[::-1][:3]
        parts = [f"{int(d)}: {probs[d] * 100:.1f}%" for d in top]
        self.detail_var.set("  ·  ".join(parts))

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z)
        e = np.exp(np.clip(z, -500, 500))
        return e / np.sum(e)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    if not os.path.isfile(WEIGHTS_PATH):
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Trained model required",
            "No trained weights found.\n\n"
            f"Expected file:\n{WEIGHTS_PATH}\n\n"
            "Train the network first (creates mnist_model.npz):\n"
            "  python train_mnist.py\n\n"
            "Then run this app again.",
        )
        root.destroy()
        sys.exit(1)

    app = DigitApp()
    app.run()


if __name__ == "__main__":
    main()
