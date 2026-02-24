from __future__ import annotations

import numpy as np

from visualization.stats.aggregate import mean_std, mean_std_matrix


def test_mean_std_filters_positive() -> None:
    s = mean_std(np.array([-1.0, 0.0, 1.0, 3.0]), filter_positive=True)
    assert np.isclose(s.mean, 2.0)
    assert np.isclose(s.std, 1.0)
    assert s.count == 2


def test_mean_std_matrix_shape() -> None:
    m = mean_std_matrix([np.array([1.0, 2.0]), np.array([3.0, 5.0])])
    assert m.shape == (2, 2)
