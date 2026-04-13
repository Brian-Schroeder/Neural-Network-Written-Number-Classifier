"""Neural network architecture visualization."""

from __future__ import annotations

from typing import List, Tuple, Any

_PALETTE = {
    "input": "#2563EB",
    "Dense": "#059669",
    "Dense + ReLU": "#D97706",
    "Dense + Sigmoid": "#7C3AED",
    "Dense + Softmax": "#0891B2",
}


def extract_architecture(model: Any) -> List[Tuple[int, str]]:
    """Extract layer sizes and types from a Sequential model."""
    stages: List[Tuple[int, str]] = []

    for layer in model.layers:
        name = type(layer).__name__

        if name == "Dense":
            in_f = layer.in_features
            out_f = layer.out_features
            if not stages:
                stages.append((in_f, "input"))
            stages.append((out_f, "Dense"))
        elif name in ("ReLU", "Sigmoid", "Softmax") and stages:
            prev_size, prev_type = stages[-1]
            if prev_type != "input":
                stages[-1] = (prev_size, f"{prev_type} + {name}")

    return stages


def _layout_nodes(
    stages: List[Tuple[int, str]],
    width: int,
    height: int,
) -> Tuple[List[List[Tuple[float, float]]], int]:
    """Compute pixel positions for each node. Returns (positions, node_radius_px)."""
    n_stages = len(stages)
    pad_x = 72
    pad_y = 88
    label_h = 56
    draw_h = height - pad_y * 2 - label_h

    max_n = max(s[0] for s in stages)
    min_spacing = 22
    node_r = min(14, max(8, int((draw_h - (max_n - 1) * min_spacing) / (2 * max_n))))
    neuron_spacing = max(min_spacing, (draw_h - 2 * node_r) / max(max_n - 1, 1))

    col_w = (width - 2 * pad_x) / max(n_stages - 1, 1)
    positions: List[List[Tuple[float, float]]] = []

    for i, (num_neurons, _) in enumerate(stages):
        cx = pad_x + i * col_w
        total_h = (num_neurons - 1) * neuron_spacing
        y0 = pad_y + (draw_h - total_h) / 2
        col = []
        for j in range(num_neurons):
            col.append((cx, y0 + j * neuron_spacing))
        positions.append(col)

    return positions, node_r


def visualize_model(model: Any, title: str = "Neural Network") -> None:
    """Open a native GUI window with the network diagram (tkinter, no matplotlib)."""
    import tkinter as tk
    from tkinter import font as tkfont

    stages = extract_architecture(model)
    if not stages:
        raise ValueError("Model has no layers or architecture could not be extracted.")

    root = tk.Tk()
    root.title(title)
    root.minsize(720, 480)
    root.configure(bg="#F1F5F9")

    ui = tkfont.nametofont("TkDefaultFont").actual()["family"]
    title_font = tkfont.Font(family=ui, size=15, weight="bold")
    sub_font = tkfont.Font(family=ui, size=10)

    header = tk.Frame(root, bg="#F1F5F9", pady=(16, 8))
    header.pack(fill=tk.X)

    tk.Label(
        header,
        text=title,
        font=title_font,
        fg="#0F172A",
        bg="#F1F5F9",
    ).pack()
    tk.Label(
        header,
        text="Layer architecture",
        font=sub_font,
        fg="#64748B",
        bg="#F1F5F9",
    ).pack(pady=(4, 0))

    canvas_w, canvas_h = 960, 420
    cv = tk.Canvas(
        root,
        width=canvas_w,
        height=canvas_h,
        bg="#FFFFFF",
        highlightthickness=1,
        highlightbackground="#E2E8F0",
    )
    cv.pack(padx=24, pady=(0, 20))

    positions, node_r = _layout_nodes(stages, canvas_w, canvas_h)
    n_stages = len(stages)

    # Edges
    for i in range(n_stages - 1):
        for (x1, y1) in positions[i]:
            for (x2, y2) in positions[i + 1]:
                cv.create_line(x1, y1, x2, y2, fill="#E5E7EB", width=1)

    # Nodes
    for i, (pts, (_, layer_type)) in enumerate(zip(positions, stages)):
        fill = _PALETTE.get(layer_type, "#6B7280")
        for (x, y) in pts:
            cv.create_oval(
                x - node_r,
                y - node_r,
                x + node_r,
                y + node_r,
                fill=fill,
                outline="#FFFFFF",
                width=2,
            )

    # Labels under diagram
    label_font = tkfont.Font(family=ui, size=9, weight="normal")
    unit_font = tkfont.Font(family=ui, size=8)
    pad_x = 72
    col_w = (canvas_w - 2 * pad_x) / max(n_stages - 1, 1)
    label_y = canvas_h - 44

    for i, (num_neurons, layer_type) in enumerate(stages):
        cx = pad_x + i * col_w
        cv.create_text(cx, label_y, text=layer_type, font=label_font, fill="#1E293B")
        cv.create_text(cx, label_y + 18, text=f"{num_neurons} units", font=unit_font, fill="#94A3B8")

    status = tk.Frame(root, bg="#F1F5F9")
    status.pack(fill=tk.X, pady=(0, 12))
    tk.Label(
        status,
        text="Close this window to exit.",
        font=sub_font,
        fg="#94A3B8",
        bg="#F1F5F9",
    ).pack()

    root.mainloop()
