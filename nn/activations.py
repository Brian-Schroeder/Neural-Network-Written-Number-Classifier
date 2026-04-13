from __future__ import annotations

import numpy as np
from .layers import Layer


class ReLU(Layer):
    """Rectified Linear Unit activation: y = max(0, x)."""

    def __init__(self) -> None:
        self._mask: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._mask = x > 0
        return np.maximum(0.0, x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._mask is not None
        return dout * self._mask


class Sigmoid(Layer):
    """Sigmoid activation: y = 1 / (1 + exp(-x))."""

    def __init__(self) -> None:
        self._y: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        self._y = 1.0 / (1.0 + np.exp(-x))
        return self._y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._y is not None
        return dout * self._y * (1.0 - self._y)


class Softmax(Layer):
    """Softmax along last dimension."""

    def __init__(self) -> None:
        self._y: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        shift = x - np.max(x, axis=1, keepdims=True)
        ex = np.exp(np.clip(shift, -500, 500))
        self._y = ex / np.sum(ex, axis=1, keepdims=True)
        return self._y

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._y is not None
        return dout * self._y * (1.0 - self._y)
