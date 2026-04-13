from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from typing import Tuple
import numpy as np

# Official-style mirror (no TensorFlow required)
_MNIST_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_MNIST_FILES = (
    ("train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz"),
    ("train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    ("t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz"),
    ("t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
)


def _mnist_cache_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "mnist")


def _download(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        data = resp.read()
    with open(dest, "wb") as f:
        f.write(data)


def _read_images_gz(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad MNIST image magic {magic}")
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    return buf.reshape(n, rows * cols).astype(np.float64) / 255.0


def _read_labels_gz(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad MNIST label magic {magic}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels.astype(np.int64)


def _load_mnist_from_files() -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    cache = _mnist_cache_dir()
    paths = []
    for remote, local in _MNIST_FILES:
        url = _MNIST_BASE + remote
        dest = os.path.join(cache, local)
        if not os.path.isfile(dest):
            _download(url, dest)
        paths.append(dest)

    x_train = _read_images_gz(paths[0])
    y_train = _read_labels_gz(paths[1])
    x_test = _read_images_gz(paths[2])
    y_test = _read_labels_gz(paths[3])
    return (x_train, y_train), (x_test, y_test)


def load_mnist() -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """Load MNIST as flattened (N, 784) float in [0, 1] and int labels (N,).

    Tries, in order: TensorFlow Keras, torchvision, then a direct download
    (numpy + stdlib only; cached under ``data/mnist/``).
    """
    try:
        from tensorflow.keras.datasets import mnist as keras_mnist

        (x_train, y_train), (x_test, y_test) = keras_mnist.load_data()
    except ImportError:
        try:
            import torchvision.datasets as tv_datasets

            root = _mnist_cache_dir()
            train = tv_datasets.MNIST(root, train=True, download=True)
            test = tv_datasets.MNIST(root, train=False, download=True)
            x_train = train.data.numpy().reshape(-1, 784).astype(np.float64) / 255.0
            y_train = train.targets.numpy().astype(np.int64)
            x_test = test.data.numpy().reshape(-1, 784).astype(np.float64) / 255.0
            y_test = test.targets.numpy().astype(np.int64)
        except (ImportError, OSError):
            (x_train, y_train), (x_test, y_test) = _load_mnist_from_files()
    else:
        x_train = x_train.reshape(-1, 784).astype(np.float64) / 255.0
        x_test = x_test.reshape(-1, 784).astype(np.float64) / 255.0
        y_train = y_train.astype(np.int64)
        y_test = y_test.astype(np.int64)

    return (x_train, y_train), (x_test, y_test)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle and split: train (first 1-test_size), test (last test_size)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X, y = X[idx], y[idx]
    n_test = int(len(X) * test_size)
    return X[n_test:], y[n_test:], X[:n_test], y[:n_test]
