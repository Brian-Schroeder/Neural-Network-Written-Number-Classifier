from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np


ParamGrad = Tuple[np.ndarray, np.ndarray]


class Optimizer:
    """Base optimizer interface."""

    def step(self, params_and_grads: Iterable[ParamGrad]) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional momentum."""

    def __init__(self, lr: float = 1e-2, momentum: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self._velocities: dict[int, np.ndarray] = {}

    def step(self, params_and_grads: Iterable[ParamGrad]) -> None:
        for param, grad in params_and_grads:
            if self.momentum == 0.0:
                param -= self.lr * grad
            else:
                v = self._velocities.setdefault(id(param), np.zeros_like(param))
                v[:] = self.momentum * v - self.lr * grad
                param += v
