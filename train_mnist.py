#!/usr/bin/env python3
"""Train a small MLP on MNIST for digit classification; saves weights for digit_gui.py."""

from __future__ import annotations

import os
import numpy as np

from nn import Sequential, Dense, ReLU, CrossEntropyLoss, SGD
from nn.data import load_mnist, train_test_split
from nn.persist import save_weights

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "mnist_model.npz")


def build_model(seed: int = 42) -> Sequential:
    return Sequential(
        [
            Dense(784, 128, seed=seed),
            ReLU(),
            Dense(128, 64, seed=seed + 1),
            ReLU(),
            Dense(64, 10, seed=seed + 2),
        ]
    )


def main() -> None:
    (x_train, y_train), (x_test, y_test) = load_mnist()
    x_train, y_train, x_val, y_val = train_test_split(x_train, y_train, test_size=0.1, seed=42)

    model = build_model()
    loss_fn = CrossEntropyLoss()
    optim = SGD(lr=0.1, momentum=0.0)

    epochs = 15
    batch_size = 128
    n = len(x_train)

    for epoch in range(1, epochs + 1):
        idx = np.random.permutation(n)
        x_train, y_train = x_train[idx], y_train[idx]
        total_loss = 0.0
        batches = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = x_train[start:end]
            yb = y_train[start:end]

            logits = model.forward(xb)
            loss = loss_fn.forward(logits, yb)
            dlogits = loss_fn.backward()
            model.backward(dlogits)
            optim.step(model.parameters())

            total_loss += loss
            batches += 1

        logits_val = model.forward(x_val)
        pred_val = np.argmax(logits_val, axis=1)
        acc_val = float((pred_val == y_val).mean())
        logits_te = model.forward(x_test)
        pred_te = np.argmax(logits_te, axis=1)
        acc_te = float((pred_te == y_test).mean())
        print(
            f"epoch {epoch:02d}  loss={total_loss / max(batches, 1):.4f}  "
            f"val_acc={acc_val:.4f}  test_acc={acc_te:.4f}"
        )

    save_weights(model, WEIGHTS_PATH)
    print(f"Saved weights to {WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
