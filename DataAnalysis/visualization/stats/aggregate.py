from __future__ import annotations

import numpy as np

from visualization.types import SummaryStats


def mean_std(arr, *, filter_positive: bool = False, ignore_nan: bool = True) -> SummaryStats:
    a = np.asarray(arr, dtype=float)
    if filter_positive:
        a = a[a > 0]
    if ignore_nan:
        a = a[~np.isnan(a)]
    if a.size == 0:
        return SummaryStats(mean=float("nan"), std=float("nan"), count=0)
    return SummaryStats(mean=float(np.mean(a)), std=float(np.std(a)), count=int(a.size))


def mean_std_matrix(arr_list: list[np.ndarray], *, filter_positive: bool = False) -> np.ndarray:
    out = []
    for arr in arr_list:
        s = mean_std(arr, filter_positive=filter_positive)
        out.append((s.mean, s.std))
    return np.asarray(out, dtype=float)
