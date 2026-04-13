from __future__ import annotations

from typing import List, Tuple

from .layers import Layer


class Sequential:
    """Simple sequential container of layers."""

    def __init__(self, layers: List[Layer] | None = None) -> None:
        self.layers: List[Layer] = layers or []

    def add(self, layer: Layer) -> None:
        self.layers.append(layer)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def parameters(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        for layer in self.layers:
            out.extend(layer.params())
        return out
