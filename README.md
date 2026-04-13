# DNN — NumPy neural networks from scratch

Educational project: a small deep-learning stack implemented in **NumPy** (dense layers, ReLU, losses, SGD), plus **MNIST** training and a **tkinter** app to draw digits **0–9** and see live predictions.

## Features

- **`nn/`** — `Sequential` model, `Dense`, activations (`ReLU`, `Sigmoid`, `Softmax`), `MSELoss`, `CrossEntropyLoss`, `SGD`, weight save/load (`.npz`)
- **`train_mnist.py`** — trains an MLP on MNIST and writes **`mnist_model.npz`**
- **`digit_gui.py`** — draw a digit with the mouse; the network predicts the class
- **`visualize_gui.py`** — shows the MNIST classifier architecture (tkinter)

## Requirements

- Python 3.10+ recommended  
- Core training/inference for the bundled scripts: **NumPy** (and **tkinter**, usually included with Python on Windows)

```bash
pip install -r requirements.txt
```

MNIST is loaded without TensorFlow if needed: the code tries TensorFlow Keras, then **torchvision**, then a **direct download** (cached under `data/mnist/`). You do **not** need TensorFlow installed to train.

## Quick start

Clone the repo, create a virtual environment, install dependencies, then train and run the GUI from the project root:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python train_mnist.py
python digit_gui.py
```

- **`train_mnist.py`** — creates `mnist_model.npz` next to the scripts (you can add it to `.gitignore` if you do not want to commit weights).
- **`digit_gui.py`** — requires `mnist_model.npz`; if it is missing, the app tells you to run training first.

## Screenshots

Examples from **`digit_gui.py`** (draw a digit, then the model predicts the class):

|  |  |
|:--:|:--:|
| **0** | **1** |
| ![Digit 0](<screenshots/0 digit.png>) | ![Digit 1](<screenshots/1 digit.png>) |
| **2** | **9** |
| ![Digit 2](<screenshots/2 digit.png>) | ![Digit 9](<screenshots/9 digit.png>) |

Paths use angle brackets because the filenames contain spaces.

## Project layout

| Path | Role |
|------|------|
| `screenshots/` | README images (your screenshots go here) |
| `nn/layers.py` | `Dense` and base `Layer` |
| `nn/activations.py` | `ReLU`, `Sigmoid`, `Softmax` |
| `nn/losses.py` | `MSELoss`, `CrossEntropyLoss` |
| `nn/optimizers.py` | `SGD` (+ optional momentum) |
| `nn/model.py` | `Sequential` |
| `nn/data.py` | MNIST loading, `train_test_split` |
| `nn/persist.py` | `save_weights` / `load_weights` |
| `nn/visualizer.py` | Architecture diagram helper |
