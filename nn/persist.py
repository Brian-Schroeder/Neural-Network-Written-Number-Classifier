"""Save and load Dense layer weights for a Sequential model."""

from __future__ import annotations

from typing import Any
import numpy as np

from .layers import Dense


def save_weights(model: Any, path: str) -> None:
    """Save all Dense W, b arrays to an .npz file."""
    data: dict[str, np.ndarray] = {}
    i = 0
    for layer in model.layers:
        if isinstance(layer, Dense):
            data[f"W{i}"] = np.array(layer.W, copy=True)
            if layer.b is not None:
                data[f"b{i}"] = np.array(layer.b, copy=True)
            i += 1
    np.savez(path, **data)


def load_weights(model: Any, path: str) -> None:
    """Load weights into Dense layers (same architecture as when saved)."""
    z = np.load(path)
    i = 0
    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.W[...] = z[f"W{i}"]
            if layer.b is not None:
                layer.b[...] = z[f"b{i}"]
            layer.dW.fill(0.0)
            if layer.db is not None:
                layer.db.fill(0.0)
            i += 1
