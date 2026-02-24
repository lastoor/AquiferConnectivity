from __future__ import annotations

import numpy as np

from visualization.metrics.geometry import distance_distribution, endpoint_distance, resample_by_arclength


def test_resample_by_arclength_preserves_endpoints() -> None:
    p = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    r = resample_by_arclength(p, ds=0.5)
    assert np.allclose(r[0], p[0])
    assert np.allclose(r[-1], p[-1])


def test_distance_distribution_non_negative() -> None:
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 1.0], [1.0, 1.0]])
    d, dab, dba = distance_distribution(a, b, ds=0.5)
    assert d.size == dab.size + dba.size
    assert np.all(d >= 0)


def test_endpoint_distance_signed_and_abs() -> None:
    a = np.array([[0.0, 0.0], [1.0, 5.0]])
    b = np.array([[0.0, 0.0], [1.0, 3.0]])
    assert endpoint_distance(a, b, signed=True) == 2.0
    assert endpoint_distance(b, a, signed=False) == 2.0
