from __future__ import annotations

from typing import List, Tuple
import numpy as np


class Layer:
    """Base layer interface."""

    def forward(self, x: np.ndarray) -> np.ndarray:  # noqa: D401
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:  # noqa: D401
        raise NotImplementedError

    def params(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        return []


class Dense(Layer):
    """Fully connected layer: y = x @ W + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True, seed: int | None = 42):
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        if seed is not None:
            np.random.seed(seed)
        scale = np.sqrt(2.0 / in_features)
        self.W: np.ndarray = np.random.randn(in_features, out_features).astype(np.float64) * scale
        self.b: np.ndarray | None = np.zeros(out_features, dtype=np.float64) if bias else None

        self.dW: np.ndarray = np.zeros_like(self.W)
        self.db: np.ndarray | None = np.zeros_like(self.b) if self.b is not None else None

        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        out = x @ self.W
        if self.b is not None:
            out = out + self.b
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        assert self._x is not None
        self.dW = self._x.T @ dout
        if self.db is not None:
            self.db = np.sum(dout, axis=0)
        return dout @ self.W.T

    def params(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        if self.b is not None:
            return [(self.W, self.dW), (self.b, self.db)]
        return [(self.W, self.dW)]
