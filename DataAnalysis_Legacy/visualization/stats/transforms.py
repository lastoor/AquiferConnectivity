from __future__ import annotations

import numpy as np


def compute_t_ref(k_g: float, dh: float, field_size: float, length_scale: float) -> float:
    i_y = length_scale * field_size
    q_ref = k_g * dh
    return i_y / q_ref


def compute_i_y(field_size: float, length_scale: float) -> float:
    return length_scale * field_size


def compute_taylor_times(
    stats_fp: np.ndarray,
    stats_lrp: np.ndarray,
    stats_fp_dijkstra: np.ndarray,
    stats_fp_fmm: np.ndarray,
    int_fp: np.ndarray,
    int_lrp: np.ndarray,
    int_fp_dijkstra: np.ndarray,
    int_fp_fmm: np.ndarray,
    int2_fp: np.ndarray,
    int2_lrp: np.ndarray,
    int2_fp_dijkstra: np.ndarray,
    int2_fp_fmm: np.ndarray,
    dh: float,
) -> dict[str, np.ndarray]:
    taylor_lrp = stats_lrp[:, 0] - int_lrp[:, 0] + int2_lrp[:, 0] / 2.0
    taylor_fp = (stats_fp[:, 0] - int_fp[:, 0] + int2_fp[:, 0] / 2.0) / dh
    taylor_fp_dijkstra = (stats_fp_dijkstra[:, 0] - int_fp_dijkstra[:, 0] + int2_fp_dijkstra[:, 0] / 2.0) / dh
    taylor_fp_fmm = (stats_fp_fmm[:, 0] - int_fp_fmm[:, 0] + int2_fp_fmm[:, 0] / 2.0) / dh
    return {
        "lrp": taylor_lrp,
        "fp": taylor_fp,
        "fp_dijkstra": taylor_fp_dijkstra,
        "fp_fmm": taylor_fp_fmm,
    }
