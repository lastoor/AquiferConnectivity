from __future__ import annotations

import numpy as np

from visualization.metrics.line_integrals import compute_line_integral


def test_line_integral_constant_field() -> None:
    field = np.ones((10, 10), dtype=float) * 2.0
    path = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]])
    int_logk, int_logk2 = compute_line_integral(path, field, grid_spacing=1.0)

    # Path length is 7. With ds convention used in legacy script, first term has ds=0.
    assert np.isclose(int_logk, 14.0)
    assert np.isclose(int_logk2, 28.0)


def test_line_integral_none_path() -> None:
    field = np.ones((4, 4), dtype=float)
    int_logk, int_logk2 = compute_line_integral(None, field)
    assert int_logk is None
    assert int_logk2 is None
