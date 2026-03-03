from __future__ import annotations

from typing import Any

import numpy as np


def _face_to_cell_speed(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """Convert per-cell face flux arrays (qxx, qyy) to a cell-centered speed proxy."""
    qxx = np.abs(np.asarray(vx, dtype=float))
    qyy = np.abs(np.asarray(vy, dtype=float))
    if qxx.shape != qyy.shape:
        raise ValueError("face velocity components must have matching shapes")

    ny, nx = qxx.shape
    vx_cell = np.empty((ny, nx), dtype=float)
    vy_cell = np.empty((ny, nx), dtype=float)

    vx_cell[:, 0] = qxx[:, 0]
    if nx > 1:
        vx_cell[:, 1:] = 0.5 * (qxx[:, :-1] + qxx[:, 1:])

    vy_cell[0, :] = qyy[0, :]
    if ny > 1:
        vy_cell[1:, :] = 0.5 * (qyy[:-1, :] + qyy[1:, :])

    return np.sqrt(vx_cell * vx_cell + vy_cell * vy_cell)


def prepare_fmm_speed(
    vx: np.ndarray,
    vy: np.ndarray,
    layout: str = "face",
    face_mapping: str = "to_cell_avg",
    zero_policy: str = "obstacle",
    min_speed: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Build a positive speed field for FMM and a traversability mask.

    Returns
    -------
    speed : ndarray
        Positive speed array passed to FMM.
    traversable_mask : ndarray[bool]
        Cells that FMM may traverse.
    diag : dict
        Diagnostic counts and chosen policy options.
    """
    if layout not in {"face", "cell"}:
        raise ValueError("layout must be 'face' or 'cell'")
    if face_mapping not in {"to_cell_avg", "direct"}:
        raise ValueError("face_mapping must be 'to_cell_avg' or 'direct'")
    if zero_policy not in {"obstacle", "clip", "error"}:
        raise ValueError("zero_policy must be 'obstacle', 'clip', or 'error'")
    if min_speed <= 0:
        raise ValueError("min_speed must be > 0")

    vx_arr = np.asarray(vx, dtype=float)
    vy_arr = np.asarray(vy, dtype=float)
    if vx_arr.shape != vy_arr.shape:
        raise ValueError("vx and vy must have the same shape")

    if layout == "cell":
        speed_raw = np.sqrt(vx_arr * vx_arr + vy_arr * vy_arr)
    else:
        if face_mapping == "to_cell_avg":
            speed_raw = _face_to_cell_speed(vx_arr, vy_arr)
        else:
            speed_raw = np.sqrt(vx_arr * vx_arr + vy_arr * vy_arr)

    finite_mask = np.isfinite(speed_raw)
    positive_mask = speed_raw > 0
    valid_mask = finite_mask & positive_mask
    zero_or_bad_count = int((~valid_mask).sum())

    diag = {
        "layout": layout,
        "face_mapping": face_mapping,
        "zero_policy": zero_policy,
        "min_speed": float(min_speed),
        "zero_or_bad_count": zero_or_bad_count,
    }

    if zero_policy == "error":
        if zero_or_bad_count > 0:
            raise ValueError(f"Non-positive or non-finite speeds detected: {zero_or_bad_count}")
        return speed_raw, np.ones_like(speed_raw, dtype=bool), diag

    if zero_policy == "clip":
        speed = np.where(finite_mask, speed_raw, 0.0)
        speed = np.maximum(speed, min_speed)
        traversable_mask = np.ones_like(speed, dtype=bool)
        return speed, traversable_mask, diag

    # obstacle
    traversable_mask = valid_mask
    speed = np.where(valid_mask, speed_raw, min_speed)
    return speed, traversable_mask, diag

