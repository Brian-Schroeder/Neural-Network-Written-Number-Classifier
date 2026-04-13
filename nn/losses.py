from __future__ import annotations

import numpy as np


class Loss:
    """Base loss interface."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError


class MSELoss(Loss):
    """Mean Squared Error loss."""

    def __init__(self) -> None:
        self._y_pred: np.ndarray | None = None
        self._y_true: np.ndarray | None = None

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self._y_pred = y_pred
        self._y_true = y_true
        return float(np.mean((y_pred - y_true) ** 2))

    def backward(self) -> np.ndarray:
        assert self._y_pred is not None and self._y_true is not None
        batch = self._y_pred.shape[0]
        return 2.0 * (self._y_pred - self._y_true) / batch


class CrossEntropyLoss(Loss):
    """Softmax cross-entropy from logits. y_true: (batch,) int labels or (batch, C) one-hot."""

    def __init__(self) -> None:
        self._logits: np.ndarray | None = None
        self._targets: np.ndarray | None = None
        self._probs: np.ndarray | None = None

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        batch, num_classes = logits.shape
        if y_true.ndim == 2 and y_true.shape[1] == num_classes:
            targets = np.argmax(y_true, axis=1)
        else:
            targets = y_true.astype(np.int64).ravel()

        shift = logits - np.max(logits, axis=1, keepdims=True)
        ex = np.exp(np.clip(shift, -500, 500))
        sum_ex = np.sum(ex, axis=1, keepdims=True)
        log_probs = shift - np.log(sum_ex + 1e-15)
        self._logits = logits
        self._targets = targets
        self._probs = ex / sum_ex

        loss = -np.mean(log_probs[np.arange(batch), targets])
        return float(loss)

    def backward(self) -> np.ndarray:
        assert self._logits is not None and self._targets is not None and self._probs is not None
        batch = self._logits.shape[0]
        grad = self._probs.copy()
        grad[np.arange(batch), self._targets] -= 1.0
        return grad / batch
