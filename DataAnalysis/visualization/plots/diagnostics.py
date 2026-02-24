from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_ratio_fit(var: np.ndarray, slopes: np.ndarray, x1_slice: slice, x2_slice: slice):
    x1 = var[x1_slice]
    y1 = slopes[x1_slice]
    x2 = var[x2_slice]
    y2 = slopes[x2_slice]

    log_x1, log_y1 = np.log10(x1), np.log10(y1)
    log_x2, log_y2 = np.log10(x2), np.log10(y2)
    s1, i1 = np.polyfit(log_x1, log_y1, 1)
    s2, i2 = np.polyfit(log_x2, log_y2, 1)

    fig, ax = plt.subplots(figsize=[5, 4])
    ax.scatter(var, slopes, color="black", marker="o", facecolor="None")
    ax.plot(x1, 10 ** i1 * x1 ** s1, "-", label=f"slope={s1:.3f}")
    ax.plot(x2, 10 ** i2 * x2 ** s2, "-", label=f"slope={s2:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$\\sigma_{Y}^2$")
    ax.set_ylabel("$\\mathcal{T}_{\\text{min}}/\\mathcal{R}_{\\text{min}}$")
    ax.legend()
    ax.grid(True, which="both")
    fig.tight_layout()
    return fig, {"kind": "ratio_fit", "slope1": float(s1), "slope2": float(s2)}
