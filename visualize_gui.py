#!/usr/bin/env python3
"""Visualize the MNIST digit-classifier architecture."""

from __future__ import annotations

from nn import Sequential, Dense, ReLU, visualize_model


def main() -> None:
    model = Sequential(
        [
            Dense(784, 128),
            ReLU(),
            Dense(128, 64),
            ReLU(),
            Dense(64, 10),
        ]
    )
    visualize_model(model, title="MNIST digit classifier — network architecture")


if __name__ == "__main__":
    main()
