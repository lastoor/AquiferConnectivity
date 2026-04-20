from __future__ import annotations

import numpy as np

from visualization.types import PathSet


def compute_line_integral(path: np.ndarray | None, field: np.ndarray, grid_spacing: float = 1.0) -> tuple[float, float] | tuple[None, None]:
    if path is None or len(path) == 0:
        return None, None

    x_idx = np.round(path[:, 0] / grid_spacing).astype(int)
    y_idx = np.round(path[:, 1] / grid_spacing).astype(int)
    x_idx = np.clip(x_idx, 0, field.shape[0] - 1)
    y_idx = np.clip(y_idx, 0, field.shape[1] - 1)

    values = field[x_idx, y_idx]
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    ds = np.concatenate(([0.0], segment_lengths))

    int_logk = float(np.sum(values * ds))
    int_logk2 = float(np.sum((values ** 2) * ds))
    return int_logk, int_logk2


def compute_mc_line_integrals(pathset: PathSet, field: np.ndarray, grid_spacing: float = 1.0) -> dict[str, tuple[float, float] | tuple[None, None]]:
    return {
        "fp": compute_line_integral(pathset.fp, field, grid_spacing),
        "lrp": compute_line_integral(pathset.lrp, field, grid_spacing),
        "fp_dijkstra": compute_line_integral(pathset.fp_dijkstra, field, grid_spacing),
        "fp_fmm": compute_line_integral(pathset.fp_fmm, field, grid_spacing),
    }


def summarize_integral_values(values: list[float]) -> dict[str, float | list[float]]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "values": []}
    return {"mean": float(np.mean(arr)), "std": float(np.std(arr)), "values": values}
