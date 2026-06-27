"""Plotting utilities for QK-LSTM results.

Uses a non-interactive Matplotlib backend so figures can be produced in headless
environments (CI, servers) and saved to disk.
"""

from __future__ import annotations

from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from .optimizer import QKLSTMQAOAOptimizer  # noqa: E402


def visualize_results(
    optimizer: QKLSTMQAOAOptimizer, save_path: Optional[str] = None
):
    """Build a 4-panel summary figure.

    Panels: (1) training/validation loss, (2) predicted vs. baseline parameters,
    (3) baseline / predicted / refined cut values, (4) quantum kernel matrix.
    Returns the Matplotlib figure; if ``save_path`` is given it is also written.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # --- Panel 1: training dynamics -------------------------------------- #
    ax = axes[0, 0]
    if optimizer.history["train_loss"]:
        ax.plot(optimizer.history["train_loss"], label="Train Loss", linewidth=2)
        ax.plot(optimizer.history["val_loss"], label="Val Loss", linewidth=2)
        ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("QK-LSTM Training Dynamics", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Panel 2: predicted vs baseline parameters ---------------------- #
    ax = axes[0, 1]
    results = optimizer.comparison_results
    if results is not None:
        p = optimizer.qaoa_depth
        base = np.asarray(results["baseline"]["params"], dtype=float)
        pred = np.asarray(results["predicted"]["params"], dtype=float)
        ax.scatter(base[:p], base[p:], s=120, label="Baseline", marker="o")
        ax.scatter(pred[:p], pred[p:], s=120, label="Predicted", marker="x")
        ax.set_xlabel(r"$\gamma$")
        ax.set_ylabel(r"$\beta$")
        ax.legend()
    ax.set_title("QAOA Parameters: Predicted vs Baseline", fontweight="bold")
    ax.grid(True, alpha=0.3)

    # --- Panel 3: performance comparison -------------------------------- #
    ax = axes[1, 0]
    if results is not None:
        methods = ["Baseline\nQAOA", "QK-LSTM\nPrediction", "QK-LSTM +\nRefinement"]
        values = [results[k]["value"] for k in ("baseline", "predicted", "refined")]
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        bars = ax.bar(methods, values, color=colors, alpha=0.8,
                      edgecolor="black", linewidth=1.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontweight="bold")
        ax.set_ylabel("MaxCut Value")
    ax.set_title("Performance Comparison", fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # --- Panel 4: quantum kernel matrix --------------------------------- #
    ax = axes[1, 1]
    if optimizer.use_quantum and optimizer.model.quantum_kernel is not None:
        rng = np.random.default_rng(0)
        X_sample = rng.standard_normal((10, optimizer.model.input_dim))
        K = optimizer.model.quantum_kernel.compute_kernel_matrix(X_sample)
        im = ax.imshow(K, cmap="RdYlBu_r", aspect="auto", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Kernel Value")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Sample Index")
        ax.set_title("Quantum Kernel Matrix", fontweight="bold")
    else:
        ax.text(0.5, 0.5, "Quantum kernel disabled", ha="center", va="center")
        ax.set_title("Quantum Kernel Matrix", fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
    return fig
