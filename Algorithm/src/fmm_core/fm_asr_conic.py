from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fm_asr import (
    AsymmetricEuclideanNorm,
    FinslerNorm,
    _bilinear_interp,
    _nearest_grid_node,
    _segment_minimize,
    build_f_acute_mesh,
    golden_section_minimize,
    is_f_acute,
)

ArrayLike = Sequence[float]
Vec = np.ndarray


def _wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def _is_primitive(dx: int, dy: int) -> bool:
    return math.gcd(abs(dx), abs(dy)) == 1


def _solve_linear_positive(
    m: float, b: float, eps: float, lo: float, hi: float
) -> List[Tuple[float, float]]:
    """Sub-intervals of ``[lo, hi]`` where ``m*t + b > eps``."""
    if abs(m) <= 1e-18:
        return [(lo, hi)] if b > eps else []
    t0 = (eps - b) / m
    if m > 0.0:
        left = max(lo, t0)
        return [(left, hi)] if hi > left else []
    right = min(hi, t0)
    return [(lo, right)] if right > lo else []


def _solve_quadratic_positive(
    a: float, b: float, c: float, eps: float, lo: float, hi: float
) -> List[Tuple[float, float]]:
    """Sub-intervals of ``[lo, hi]`` where ``a*t^2 + b*t + c > eps``."""
    if abs(a) <= 1e-18:
        return _solve_linear_positive(b, c, eps, lo, hi)
    c_eff = c - eps
    disc = b * b - 4.0 * a * c_eff
    if disc < 0.0:
        return [(lo, hi)] if a > 0.0 else []
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    if t1 > t2:
        t1, t2 = t2, t1
    if a > 0.0:
        out: List[Tuple[float, float]] = []
        if t1 > lo:
            out.append((lo, min(t1, hi)))
        if t2 < hi:
            out.append((max(t2, lo), hi))
        return [iv for iv in out if iv[1] > iv[0]]
    left = max(lo, t1)
    right = min(hi, t2)
    return [(left, right)] if right > left else []


def _intersect_interval_lists(
    A: Sequence[Tuple[float, float]], B: Sequence[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    """Intersection of two sorted lists of disjoint intervals."""
    out: List[Tuple[float, float]] = []
    for a_lo, a_hi in A:
        for b_lo, b_hi in B:
            lo = max(a_lo, b_lo)
            hi = min(a_hi, b_hi)
            if hi > lo:
                out.append((lo, hi))
    out.sort()
    return out


@dataclass
class StrongWindZermeloNorm:
    """Conic Zermelo travel-time norm for the minimizing wind branch.

    This implements the ``Z`` branch from the arbitrary-wind Zermelo formulas,
    using a Euclidean or SPD ground metric ``G``:

        Z(v) = g(v, v) / (sqrt(lambda*g(v, v) + g(W, v)^2) + g(W, v))

    where ``lambda = 1 - g(W, W)``.  In strong and critical wind this is a conic
    norm: directions outside the admissible cone have infinite cost.
    """

    W: np.ndarray
    G: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=float))
    eps: float = 1e-12

    def __post_init__(self) -> None:
        self.W = np.asarray(self.W, dtype=float)
        self.G = np.asarray(self.G, dtype=float)
        if self.W.shape != (2,):
            raise ValueError("W must be a 2-vector")
        if self.G.shape != (2, 2):
            raise ValueError("G must be 2x2")
        if not np.allclose(self.G, self.G.T, atol=1e-12):
            raise ValueError("G must be symmetric")
        if np.min(np.linalg.eigvalsh(self.G)) <= 0.0:
            raise ValueError("G must be SPD")
        if not np.all(np.isfinite(self.W)) or not np.all(np.isfinite(self.G)):
            raise ValueError("W and G must contain only finite values")
        self.lambda_ = 1.0 - self.inner(self.W, self.W)

    def inner(self, a: ArrayLike, b: ArrayLike) -> float:
        av = np.asarray(a, dtype=float)
        bv = np.asarray(b, dtype=float)
        return float(av.T @ self.G @ bv)

    def admissibility_margin(self, u: ArrayLike) -> float:
        u = np.asarray(u, dtype=float)
        guu = self.inner(u, u)
        if guu <= self.eps:
            return 0.0
        if self.lambda_ > self.eps:
            return 1.0
        gwu = self.inner(self.W, u)
        if abs(self.lambda_) <= self.eps:
            return gwu
        disc = self.lambda_ * guu + gwu * gwu
        return min(gwu, disc)

    def is_admissible(self, u: ArrayLike, margin: Optional[float] = None) -> bool:
        threshold = self.eps if margin is None else float(margin)
        return self.admissibility_margin(u) > threshold

    def __call__(self, u: ArrayLike) -> float:
        u = np.asarray(u, dtype=float)
        guu = self.inner(u, u)
        if guu <= self.eps:
            return 0.0
        gwu = self.inner(self.W, u)

        if self.lambda_ > self.eps:
            disc = self.lambda_ * guu + gwu * gwu
            return guu / (math.sqrt(max(0.0, disc)) + gwu)

        if abs(self.lambda_) <= self.eps:
            if gwu <= self.eps:
                return float("inf")
            return guu / (2.0 * gwu)

        disc = self.lambda_ * guu + gwu * gwu
        if gwu <= self.eps or disc <= self.eps:
            return float("inf")
        return guu / (math.sqrt(disc) + gwu)

    def grad(self, u: ArrayLike) -> Vec:
        """Analytic gradient of ``Z`` on the open admissible cone.

        Raises ``NotImplementedError`` outside the cone or at the cone boundary,
        so callers (notably ``is_f_acute``) can fall through to safer paths.
        """
        u = np.asarray(u, dtype=float)
        if not self.is_admissible(u):
            raise NotImplementedError("grad undefined outside the admissible cone")
        Gu = self.G @ u
        GW = self.G @ self.W
        s = float(u @ Gu)
        if s <= self.eps:
            raise ValueError("grad undefined at the origin")
        w = float(self.W @ Gu)

        if abs(self.lambda_) <= self.eps:
            if w <= self.eps:
                raise NotImplementedError(
                    "grad undefined at the critical-wind cone boundary"
                )
            return (2.0 * Gu * w - s * GW) / (2.0 * w * w)

        D2 = self.lambda_ * s + w * w
        if D2 <= self.eps:
            raise NotImplementedError("grad undefined at the cone boundary")
        D = math.sqrt(D2)
        denom = D + w
        if denom <= self.eps:
            raise NotImplementedError("grad undefined: denominator near zero")
        d_D = (self.lambda_ * Gu + w * GW) / D
        return (2.0 * Gu * denom - s * (d_D + GW)) / (denom * denom)

    def acute(self, u: ArrayLike, v: ArrayLike) -> bool:
        """Smooth-gradient F-acute predicate restricted to the admissible cone.

        Returns ``False`` if either ``u`` or ``v`` is outside (or on) the open
        admissible cone where ``Z`` is non-smooth; otherwise applies the smooth
        criterion ``<u, grad Z(v)> >= 0`` and ``<v, grad Z(u)> >= 0``.
        """
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        if not self.is_admissible(u) or not self.is_admissible(v):
            return False
        try:
            gu = self.grad(u)
            gv = self.grad(v)
        except NotImplementedError:
            return False
        return float(u @ gv) >= -self.eps and float(v @ gu) >= -self.eps

    def admissible_intervals(
        self,
        x: ArrayLike,
        a: ArrayLike,
        b: ArrayLike,
        *,
        eps: Optional[float] = None,
    ) -> List[Tuple[float, float]]:
        """Analytic admissible t-intervals in ``[0,1]`` for ``u(t)=(1-t)a+tb-x``.

        The cone admissibility conditions are at most a linear and a quadratic
        constraint in ``t``, so the admissible set can be computed in closed
        form by intersecting their positive sets with ``[0,1]``.  By
        Lorentz-cone convexity the final set is a single interval (or empty),
        but this routine returns a list for uniformity with the helpers.
        """
        margin = self.eps if eps is None else float(eps)
        x_arr = np.asarray(x, dtype=float)
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)
        r = a_arr - x_arr
        s = b_arr - a_arr

        w0 = self.inner(self.W, r)
        w1 = self.inner(self.W, s)
        q0 = self.inner(r, r)
        q1 = self.inner(r, s)
        q2 = self.inner(s, s)

        if self.lambda_ > self.eps:
            # Subcritical: cone is the full punctured plane; only need u(t) != 0.
            return _solve_quadratic_positive(q2, 2.0 * q1, q0, margin, 0.0, 1.0)

        linear_set = _solve_linear_positive(w1, w0, margin, 0.0, 1.0)
        if not linear_set:
            return []

        if abs(self.lambda_) <= self.eps:
            return linear_set

        a2 = self.lambda_ * q2 + w1 * w1
        a1 = 2.0 * (self.lambda_ * q1 + w0 * w1)
        a0 = self.lambda_ * q0 + w0 * w0
        quadratic_set = _solve_quadratic_positive(a2, a1, a0, margin, 0.0, 1.0)
        if not quadratic_set:
            return []

        return _intersect_interval_lists(linear_set, quadratic_set)

    def edge_update(
        self,
        x: ArrayLike,
        a: ArrayLike,
        b: ArrayLike,
        da: float,
        db: float,
        *,
        samples: int = 64,
        tol: float = 1e-10,
    ) -> Optional[float]:
        result = self.edge_argmin(x, a, b, da, db, samples=samples, tol=tol)
        if result is None:
            return None
        return result[2]

    def edge_argmin(
        self,
        x: ArrayLike,
        a: ArrayLike,
        b: ArrayLike,
        da: float,
        db: float,
        *,
        samples: int = 64,
        tol: float = 1e-10,
    ) -> Optional[Tuple[float, np.ndarray, float]]:
        del samples  # kept for API compatibility; analytic admissibility used.
        x_arr = np.asarray(x, dtype=float)
        a_arr = np.asarray(a, dtype=float)
        b_arr = np.asarray(b, dtype=float)

        def direction(t: float) -> np.ndarray:
            return (1.0 - t) * a_arr + t * b_arr - x_arr

        def phi(t: float) -> float:
            cost = float(self(direction(t)))
            if not np.isfinite(cost):
                return float("inf")
            return cost + (1.0 - t) * float(da) + t * float(db)

        intervals = self.admissible_intervals(x_arr, a_arr, b_arr)
        if not intervals:
            return None

        best_t: Optional[float] = None
        best_val = float("inf")
        for left, right in intervals:
            if right <= left:
                continue
            candidates = [left, right]
            if right - left > tol:
                t_mid, val_mid = golden_section_minimize(
                    phi, left, right, tol=tol
                )
                candidates.append(t_mid)
                if val_mid < best_val:
                    best_t = t_mid
                    best_val = val_mid
            for t in candidates:
                val = phi(t)
                if val < best_val:
                    best_t = t
                    best_val = val

        if best_t is None or not np.isfinite(best_val):
            return None
        q = (1.0 - best_t) * a_arr + best_t * b_arr
        return best_t, q, float(best_val)


@dataclass
class ConicStencil:
    offsets: List[Tuple[int, int]]
    is_closed: bool = False

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return [(idx, idx + 1) for idx in range(len(self.offsets) - 1)]


@dataclass
class MixedStencil:
    """Per-cell stencil that may be a closed cyclic F-acute mesh or an open cone."""

    offsets: List[Tuple[int, int]]
    is_closed: bool

    @property
    def edges(self) -> List[Tuple[int, int]]:
        n = len(self.offsets)
        if self.is_closed:
            return [(idx, (idx + 1) % n) for idx in range(n)]
        return [(idx, idx + 1) for idx in range(n - 1)]


def build_conic_stencil(
    F: StrongWindZermeloNorm,
    radius: int = 8,
    *,
    require_acute: bool = True,
) -> List[Tuple[int, int]]:
    """Build a sorted admissible integer stencil for a conic norm.

    When ``require_acute`` is true, applies cone-aware Stern-Brocot refinement
    starting from the angular-extreme admissible primitive offsets within
    ``radius``.  Adjacent vertices are accepted whenever they are F-acute under
    ``F``; otherwise the primitive part of their mediant ``u+v`` is inserted
    and the pair is reprocessed.  Refinement bails out (accepting the current
    pair as best effort) when the mediant exceeds ``radius`` or coincides with
    one of the endpoints.  This mirrors ``build_f_acute_mesh`` in ``fm_asr``
    but restricted to the open admissible cone, which is convex by
    Lorentz-cone geometry so mediants stay admissible by construction.
    """
    radius = int(radius)
    if radius < 1:
        raise ValueError("radius must be >= 1")

    admissible: List[Tuple[int, int]] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            if not _is_primitive(dx, dy):
                continue
            u = np.array([dx, dy], dtype=float)
            if F.is_admissible(u):
                admissible.append((dx, dy))

    if len(admissible) < 2:
        raise RuntimeError("conic stencil has fewer than two admissible offsets")

    if F.lambda_ > F.eps:
        admissible.sort(key=lambda item: math.atan2(item[1], item[0]))
    else:
        center_angle = math.atan2(float(F.W[1]), float(F.W[0]))
        admissible.sort(
            key=lambda item: _wrap_angle(
                math.atan2(item[1], item[0]) - center_angle
            )
        )

    if not require_acute or len(admissible) <= 2:
        return admissible

    # Seed the refinement with the full sorted admissible list so the
    # Stern-Brocot loop only fills *inner* gaps where the smooth F-acute
    # criterion fails between consecutive offsets.  This mirrors how
    # ``build_f_acute_mesh`` seeds the closed Randers mesh with the four
    # cardinal directions before refining.
    L: List[Tuple[int, int]] = [admissible[0]]
    pending: List[Tuple[int, int]] = list(reversed(admissible[1:]))
    max_iter = 32 * radius * radius + 256

    iters = 0
    while pending and iters < max_iter:
        iters += 1
        u_int = L[-1]
        v_int = pending[-1]
        u_vec = np.asarray(u_int, dtype=float)
        v_vec = np.asarray(v_int, dtype=float)
        if is_f_acute(u_vec, v_vec, F):
            L.append(pending.pop())
            continue
        w_dx = u_int[0] + v_int[0]
        w_dy = u_int[1] + v_int[1]
        # Mediants of non-Stern-Brocot neighbors can pick up a common factor;
        # reduce to the primitive representative before refining further.
        g = math.gcd(abs(w_dx), abs(w_dy)) or 1
        w_int = (w_dx // g, w_dy // g)
        if (
            max(abs(w_int[0]), abs(w_int[1])) > radius
            or w_int == u_int
            or w_int == v_int
            or not F.is_admissible(np.asarray(w_int, dtype=float))
        ):
            # Radius cap: accept the non-acute pair as best effort.
            L.append(pending.pop())
            continue
        pending.append(w_int)

    while pending:
        L.append(pending.pop())

    seen: set = set()
    deduped: List[Tuple[int, int]] = []
    for offset in L:
        if offset in seen:
            continue
        seen.add(offset)
        deduped.append(offset)
    return deduped


def dispersion_tensor_field_from_velocity(
    W_field: np.ndarray,
    *,
    molecular_diffusion: float = 1e-9,
    alpha_l: float = 0.1,
    alpha_t: float = 0.01,
    velocity_floor: Optional[float] = None,
) -> np.ndarray:
    """Compute the cellwise 2D PyPar2 dispersion tensor field.

    The formula matches ``FlowSimulation.simulation.particle_tracking.pypar2``:
    longitudinal dispersion is ``alpha_l * |v| + Dm`` and transverse dispersion
    is ``alpha_t * |v| + Dm`` in the local flow-aligned frame.
    """
    W = np.asarray(W_field, dtype=float)
    if W.ndim != 3 or W.shape[-1] != 2:
        raise ValueError("W_field must have shape (ny, nx, 2)")
    if not np.all(np.isfinite(W)):
        raise ValueError("W_field must contain only finite values")
    dm = float(molecular_diffusion)
    alpha_l = float(alpha_l)
    alpha_t = float(alpha_t)
    if dm < 0.0:
        raise ValueError("molecular_diffusion must be nonnegative")
    if alpha_l < 0.0 or alpha_t < 0.0:
        raise ValueError("alpha_l and alpha_t must be nonnegative")
    if velocity_floor is None:
        velocity_floor_value = 0.01 * dm / alpha_l if alpha_l > 0.0 else 0.0
    else:
        velocity_floor_value = float(velocity_floor)
    if velocity_floor_value < 0.0:
        raise ValueError("velocity_floor must be nonnegative")

    ny, nx = W.shape[:2]
    D = np.zeros((ny, nx, 2, 2), dtype=float)
    for j in range(ny):
        for i in range(nx):
            vx = float(W[j, i, 0])
            vy = float(W[j, i, 1])
            vnorm = math.hypot(vx, vy)
            if velocity_floor_value > 0.0 and vnorm < velocity_floor_value:
                vx = velocity_floor_value
                vy = 0.0
                vnorm = velocity_floor_value
            if vnorm <= 1e-12:
                D[j, i] = np.eye(2, dtype=float) * dm
                continue
            common = alpha_t * vnorm + dm
            beta = alpha_l - alpha_t
            D[j, i] = np.array(
                [
                    [common + beta * vx * vx / vnorm, beta * vx * vy / vnorm],
                    [beta * vx * vy / vnorm, common + beta * vy * vy / vnorm],
                ],
                dtype=float,
            )
    return D


def self_metric_from_dispersion_velocity(
    W_field: np.ndarray,
    *,
    dt: float,
    molecular_diffusion: float = 0.0,
    alpha_l: float = 0.0,
    alpha_t: float = 0.0,
    velocity_floor: Optional[float] = None,
    eigenvalue_floor: float = 1e-12,
) -> np.ndarray:
    """Return ``G = inv(2D/dt)`` for the dispersion step-velocity ellipse."""
    dt = float(dt)
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be positive and finite")
    eigenvalue_floor = float(eigenvalue_floor)
    if eigenvalue_floor <= 0.0 or not math.isfinite(eigenvalue_floor):
        raise ValueError("eigenvalue_floor must be positive and finite")

    D = dispersion_tensor_field_from_velocity(
        W_field,
        molecular_diffusion=molecular_diffusion,
        alpha_l=alpha_l,
        alpha_t=alpha_t,
        velocity_floor=velocity_floor,
    )
    V = (2.0 / dt) * D
    ny, nx = V.shape[:2]
    G = np.zeros_like(V)
    for j in range(ny):
        for i in range(nx):
            eigvals, eigvecs = np.linalg.eigh(V[j, i])
            eigvals = np.maximum(eigvals, eigenvalue_floor)
            G[j, i] = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
    return G


def build_strong_wind_metric_from_W(
    W_field: np.ndarray,
    *,
    direction: str = "from_source",
    min_speed: float = 1.0 + 1e-9,
) -> Callable[[float, float], StrongWindZermeloNorm]:
    """Return an index-based metric factory for a strong-wind vector field.

    This helper is mainly useful for simple unit-spaced grids.  For grids with a
    nonzero origin or nonunit spacing, prefer ``ConicFMASR2D.from_velocity_field``.
    """
    W = np.asarray(W_field, dtype=float)
    if W.ndim != 3 or W.shape[-1] != 2:
        raise ValueError("W_field must have shape (ny, nx, 2)")
    if not np.all(np.isfinite(W)):
        raise ValueError("W_field must contain only finite values")
    if direction not in {"from_source", "to_source"}:
        raise ValueError("direction must be 'from_source' or 'to_source'")
    sign = -1.0 if direction == "from_source" else 1.0
    ny, nx = W.shape[:2]

    def metric_at(x: float, y: float) -> StrongWindZermeloNorm:
        i = min(nx - 1, max(0, int(round(x))))
        j = min(ny - 1, max(0, int(round(y))))
        wind = sign * W[j, i]
        speed = float(np.linalg.norm(wind))
        if speed < min_speed:
            if speed <= 0.0:
                wind = np.array([sign * min_speed, 0.0], dtype=float)
            else:
                wind = wind * (float(min_speed) / speed)
        return StrongWindZermeloNorm(wind)

    return metric_at


@dataclass
class ConicFMASR2D:
    """Experimental FM-ASR solver for conic strong-wind Zermelo metrics."""

    shape: Tuple[int, int]
    h: float
    metric_at: Callable[[float, float], StrongWindZermeloNorm]
    origin: Tuple[float, float] = (0.0, 0.0)
    stencil_radius: int = 8
    edge_samples: int = 64
    edge_tol: float = 1e-10

    _stencil_cache: Dict[Tuple[int, int], ConicStencil] = field(init=False, repr=False)
    _reverse_neighbors: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.ny, self.nx = self.shape
        if self.ny <= 0 or self.nx <= 0:
            raise ValueError("shape must be positive")
        if self.h <= 0.0:
            raise ValueError("h must be positive")
        self._stencil_cache = {}
        self._reverse_neighbors = {}
        self._build_stencil_graph()

    @classmethod
    def from_velocity_field(
        cls,
        W_field: np.ndarray,
        h: float,
        *,
        G_field: Optional[np.ndarray] = None,
        origin: Tuple[float, float] = (0.0, 0.0),
        direction: str = "from_source",
        min_speed: float = 1.0 + 1e-9,
        min_wind_factor: Optional[float] = None,
        stencil_radius: int = 8,
        edge_samples: int = 64,
        edge_tol: float = 1e-10,
    ) -> "ConicFMASR2D":
        W = np.asarray(W_field, dtype=float)
        if W.ndim != 3 or W.shape[-1] != 2:
            raise ValueError("W_field must have shape (ny, nx, 2)")
        if not np.all(np.isfinite(W)):
            raise ValueError("W_field must contain only finite values")
        if direction not in {"from_source", "to_source"}:
            raise ValueError("direction must be 'from_source' or 'to_source'")
        if min_speed <= 1.0:
            raise ValueError("min_speed must be > 1 for strong-wind conic metrics")
        if min_wind_factor is None:
            min_wind_factor = min_speed
        min_wind_factor = float(min_wind_factor)
        if min_wind_factor <= 1.0:
            raise ValueError("min_wind_factor must be > 1 for strong-wind conic metrics")
        if G_field is not None:
            G = np.asarray(G_field, dtype=float)
            if G.shape != W.shape[:2] + (2, 2):
                raise ValueError("G_field must have shape (ny, nx, 2, 2)")
            if not np.all(np.isfinite(G)):
                raise ValueError("G_field must contain only finite values")
            for j in range(G.shape[0]):
                for i in range(G.shape[1]):
                    if not np.allclose(G[j, i], G[j, i].T, atol=1e-12):
                        raise ValueError("G_field entries must be symmetric")
                    if np.min(np.linalg.eigvalsh(G[j, i])) <= 0.0:
                        raise ValueError("G_field entries must be SPD")
        else:
            G = None

        sign = -1.0 if direction == "from_source" else 1.0
        ny, nx = W.shape[:2]
        ox, oy = origin

        def metric_at(x: float, y: float) -> StrongWindZermeloNorm:
            i = int(round((x - ox) / h))
            j = int(round((y - oy) / h))
            i = min(nx - 1, max(0, i))
            j = min(ny - 1, max(0, j))
            wind = sign * W[j, i]
            G_local = None if G is None else G[j, i]
            if G_local is None:
                speed = float(np.linalg.norm(wind))
                threshold = float(min_speed)
            else:
                speed = math.sqrt(max(0.0, float(wind.T @ G_local @ wind)))
                threshold = float(min_wind_factor)
            if speed < threshold:
                if speed <= 0.0:
                    if G_local is None:
                        wind = np.array([sign * threshold, 0.0], dtype=float)
                    else:
                        axis = np.array([sign, 0.0], dtype=float)
                        axis_norm = math.sqrt(max(0.0, float(axis.T @ G_local @ axis)))
                        if axis_norm <= 0.0:
                            axis = np.array([0.0, sign], dtype=float)
                            axis_norm = math.sqrt(max(0.0, float(axis.T @ G_local @ axis)))
                        wind = axis * (threshold / axis_norm)
                else:
                    wind = wind * (threshold / speed)
            if G_local is None:
                return StrongWindZermeloNorm(wind)
            return StrongWindZermeloNorm(wind, G=G_local)

        return cls(
            shape=(ny, nx),
            h=h,
            metric_at=metric_at,
            origin=origin,
            stencil_radius=stencil_radius,
            edge_samples=edge_samples,
            edge_tol=edge_tol,
        )

    def _is_inside(self, node: Tuple[int, int]) -> bool:
        j, i = node
        return 0 <= i < self.nx and 0 <= j < self.ny

    def _cell_position(self, j: int, i: int) -> np.ndarray:
        ox, oy = self.origin
        return np.array([ox + i * self.h, oy + j * self.h], dtype=float)

    def _metric_at_cell(self, j: int, i: int) -> StrongWindZermeloNorm:
        x, y = self._cell_position(j, i)
        return self.metric_at(float(x), float(y))

    def stencil_at(self, j: int, i: int) -> ConicStencil:
        key = (j, i)
        stencil = self._stencil_cache.get(key)
        if stencil is None:
            offsets = build_conic_stencil(
                self._metric_at_cell(j, i),
                radius=self.stencil_radius,
                require_acute=True,
            )
            stencil = ConicStencil(offsets)
            self._stencil_cache[key] = stencil
        return stencil

    def _build_stencil_graph(self) -> None:
        for j in range(self.ny):
            for i in range(self.nx):
                stencil = self.stencil_at(j, i)
                for idx, (dx, dy) in enumerate(stencil.offsets):
                    y_node = (j + dy, i + dx)
                    if self._is_inside(y_node):
                        self._reverse_neighbors.setdefault(y_node, []).append((j, i, idx))

    def solve(self, source_mask: np.ndarray) -> np.ndarray:
        return self.solve_source_mask(source_mask)

    def solve_source_mask(self, source_mask: np.ndarray) -> np.ndarray:
        source_mask = np.asarray(source_mask, dtype=bool)
        if source_mask.shape != self.shape:
            raise ValueError("source_mask has wrong shape")
        if not np.any(source_mask):
            raise ValueError("source_mask is empty")

        d = np.full(self.shape, np.inf, dtype=float)
        accepted = np.zeros(self.shape, dtype=bool)
        heap: List[Tuple[float, Tuple[int, int]]] = []
        for j, i in np.argwhere(source_mask):
            d[int(j), int(i)] = 0.0
            heapq.heappush(heap, (0.0, (int(j), int(i))))

        while heap:
            dist, (j, i) = heapq.heappop(heap)
            if accepted[j, i] or dist != d[j, i]:
                continue
            accepted[j, i] = True
            y_pos = self._cell_position(j, i)

            for xj, xi, local_idx in self._reverse_neighbors.get((j, i), []):
                if accepted[xj, xi]:
                    continue
                x_pos = self._cell_position(xj, xi)
                F_x = self._metric_at_cell(xj, xi)
                stencil = self.stencil_at(xj, xi)
                best = d[xj, xi]

                direct_cost = float(F_x(y_pos - x_pos))
                if np.isfinite(direct_cost):
                    cand = direct_cost + d[j, i]
                    if cand < best:
                        best = cand

                for neighbor_idx in (local_idx - 1, local_idx + 1):
                    if neighbor_idx < 0 or neighbor_idx >= len(stencil.offsets):
                        continue
                    dx, dy = stencil.offsets[neighbor_idx]
                    zj = xj + dy
                    zi = xi + dx
                    if not (0 <= zi < self.nx and 0 <= zj < self.ny):
                        continue
                    if not accepted[zj, zi]:
                        continue
                    z_pos = self._cell_position(zj, zi)
                    cand_edge = F_x.edge_update(
                        x_pos,
                        z_pos,
                        y_pos,
                        float(d[zj, zi]),
                        float(d[j, i]),
                        samples=self.edge_samples,
                        tol=self.edge_tol,
                    )
                    if cand_edge is not None and cand_edge < best:
                        best = float(cand_edge)

                if best + 1e-15 < d[xj, xi]:
                    d[xj, xi] = best
                    heapq.heappush(heap, (best, (xj, xi)))

        return d


def _edge_pairs(offsets: Sequence[Tuple[int, int]], is_closed: bool) -> List[Tuple[int, int]]:
    n = len(offsets)
    if n < 2:
        return []
    if is_closed:
        return [(k, (k + 1) % n) for k in range(n)]
    return [(k, k + 1) for k in range(n - 1)]


def _safe_interp(
    dist: np.ndarray,
    p: np.ndarray,
    origin: Tuple[float, float],
    h: float,
) -> float:
    """Bilinear interpolation that falls back to nearest-node on 1D strips."""
    if dist.shape[0] >= 2 and dist.shape[1] >= 2:
        return _bilinear_interp(dist, p, origin, h)
    j, i = _nearest_grid_node(p, origin, h, dist.shape)
    return float(dist[j, i])


def _trace_continuous_path(
    solver: Any,
    dist: np.ndarray,
    target_xy: ArrayLike,
    *,
    max_steps: int = 10000,
    stop_tol: float = 1e-10,
    source_mask: Optional[np.ndarray] = None,
    is_closed_per_cell: bool = False,
) -> np.ndarray:
    """Hopf-Lax continuous backtracking on a per-cell stencil.

    Mirrors ``trace_minimal_path`` in :mod:`fm_asr` but works with both the
    one-sided conic stencil and the closed Randers F-acute mesh.  ``solver``
    must expose ``stencil_at``, ``metric_at``, ``_cell_position``, ``origin``,
    ``h``, ``nx`` and ``ny`` (``ConicFMASR2D`` and ``MixedFMASR2D`` both do).
    """
    p = np.asarray(target_xy, dtype=float).copy()
    path: List[np.ndarray] = [p.copy()]
    ny, nx = dist.shape
    if source_mask is not None:
        source_mask = np.asarray(source_mask, dtype=bool)
        if source_mask.shape != dist.shape:
            raise ValueError("source_mask must match dist shape")

    for _ in range(int(max_steps)):
        d_here = _safe_interp(dist, p, solver.origin, solver.h)
        if d_here <= stop_tol:
            break
        j, i = _nearest_grid_node(p, solver.origin, solver.h, (ny, nx))
        if source_mask is not None and source_mask[j, i]:
            p = solver._cell_position(j, i)
            path[-1] = p.copy()
            break

        stencil_obj = solver.stencil_at(j, i)
        offsets = list(stencil_obj.offsets)
        cell_is_closed = (
            bool(getattr(stencil_obj, "is_closed", False))
            if not is_closed_per_cell
            else bool(getattr(stencil_obj, "is_closed", True))
        )
        pairs = _edge_pairs(offsets, cell_is_closed)
        x_pos = solver._cell_position(j, i)
        F_p = solver.metric_at(float(p[0]), float(p[1]))

        best_val = float("inf")
        best_q: Optional[np.ndarray] = None
        best_vertex_dist = float("inf")
        best_vertex: Optional[np.ndarray] = None

        # Independent single-offset vertex scan ensures we have a fallback even
        # for degenerate 1D strips where most edge-pairs leave the grid.
        for dx_off, dy_off in offsets:
            nj_off = j + dy_off
            ni_off = i + dx_off
            if not (0 <= ni_off < nx and 0 <= nj_off < ny):
                continue
            v_off = float(dist[nj_off, ni_off])
            if np.isfinite(v_off) and v_off < best_vertex_dist:
                best_vertex_dist = v_off
                best_vertex = x_pos + solver.h * np.array(
                    [dx_off, dy_off], dtype=float
                )

        for k1, k2 in pairs:
            dx1, dy1 = offsets[k1]
            dx2, dy2 = offsets[k2]
            ja, ia = j + dy1, i + dx1
            jb, ib = j + dy2, i + dx2
            if not (
                0 <= ia < nx and 0 <= ja < ny and 0 <= ib < nx and 0 <= jb < ny
            ):
                continue
            a = x_pos + solver.h * np.array([dx1, dy1], dtype=float)
            b = x_pos + solver.h * np.array([dx2, dy2], dtype=float)
            da = float(dist[ja, ia])
            db = float(dist[jb, ib])
            if np.isfinite(da) and da < best_vertex_dist:
                best_vertex_dist = da
                best_vertex = a
            if np.isfinite(db) and db < best_vertex_dist:
                best_vertex_dist = db
                best_vertex = b
            if not (np.isfinite(da) and np.isfinite(db)):
                continue
            try:
                _, q, val = _segment_minimize(F_p, p, a, b, da, db)
            except Exception:
                continue
            if np.isfinite(val) and val < best_val:
                best_val = val
                best_q = q

        if best_q is None:
            if best_vertex is None or best_vertex_dist >= d_here - 1e-14:
                break
            best_q = best_vertex
        else:
            next_dist = _safe_interp(dist, best_q, solver.origin, solver.h)
            if next_dist >= d_here - 1e-14:
                if best_vertex is None or best_vertex_dist >= d_here - 1e-14:
                    break
                best_q = best_vertex

        p = best_q
        path.append(p.copy())

    return np.asarray(path, dtype=float)


def extract_conic_fm_asr_path(
    solver: ConicFMASR2D,
    dist: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    max_steps: int = 10000,
    stop_tol: float = 1e-10,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Extract a continuous Hopf-Lax minimal path from a conic FM-ASR distance map.

    Mirrors :func:`fmm_core.fm_asr.extract_fm_asr_path` but uses the per-cell
    one-sided conic stencil and :meth:`StrongWindZermeloNorm.edge_argmin` for
    segment minimization, falling back to the lowest-distance stencil vertex
    when no admissible segment strictly decreases the interpolated distance.
    """
    dist = np.asarray(dist, dtype=float)
    source_mask = np.asarray(source_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    if source_mask.shape != dist.shape or target_mask.shape != dist.shape:
        raise ValueError("source_mask and target_mask must match dist shape")

    masked = np.where(target_mask, dist, np.inf)
    flat = int(np.argmin(masked))
    travel_value = float(masked.flat[flat])
    if not np.isfinite(travel_value):
        raise RuntimeError("No finite conic FM-ASR target travel time")
    tj, ti = (int(v) for v in np.unravel_index(flat, dist.shape))
    chosen_target = (tj, ti)
    target_xy = solver._cell_position(tj, ti)

    path_target_to_source = _trace_continuous_path(
        solver,
        dist,
        target_xy,
        max_steps=max_steps,
        stop_tol=stop_tol,
        source_mask=source_mask,
    )
    if path_target_to_source.shape[0] < 2:
        raise RuntimeError(
            "Conic FM-ASR path extraction produced fewer than two points"
        )
    return path_target_to_source[::-1], travel_value, chosen_target


def randers_from_zermelo(
    W: ArrayLike,
    G: Optional[ArrayLike] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a subcritical Zermelo pair ``(W, G)`` to Randers ``(M, beta)``.

    For ``lambda = 1 - g(W, W) > 0`` the Zermelo travel-time norm
    ``Z(v) = g(v,v) / (sqrt(lambda g(v,v) + g(W,v)^2) + g(W,v))`` admits the
    closed Randers form ``F(v) = sqrt(v^T M v) - beta^T v`` with

        M    = G / lambda + (G W)(G W)^T / lambda^2,
        beta = G W / lambda.

    The Randers validity condition ``beta^T M^{-1} beta = g(W, W) < 1`` is
    guaranteed by ``lambda > 0``.
    """
    W_arr = np.asarray(W, dtype=float)
    if W_arr.shape != (2,):
        raise ValueError("W must be a 2-vector")
    if G is None:
        G_arr = np.eye(2, dtype=float)
    else:
        G_arr = np.asarray(G, dtype=float)
        if G_arr.shape != (2, 2):
            raise ValueError("G must be 2x2")
    lambda_ = 1.0 - float(W_arr @ G_arr @ W_arr)
    if lambda_ <= 0.0:
        raise ValueError(
            "randers_from_zermelo requires subcritical wind (lambda > 0)"
        )
    GW = G_arr @ W_arr
    M = G_arr / lambda_ + np.outer(GW, GW) / (lambda_ * lambda_)
    beta = GW / lambda_
    return M, beta


@dataclass
class MixedFMASR2D:
    """FM-ASR for the Zermelo problem with per-cell wind-regime dispatch.

    Cells with subcritical wind (``lambda = 1 - g(W, W) > lambda_threshold``)
    use the equivalent Randers norm ``AsymmetricEuclideanNorm`` and a closed
    cyclic F-acute mesh built with :func:`build_f_acute_mesh`.  Cells with
    critical or supercritical wind use :class:`StrongWindZermeloNorm` with the
    one-sided conic stencil from :func:`build_conic_stencil`.  Both regimes
    share the same Dijkstra-style marching loop; the only per-cell branch is
    whether the edge updates wrap cyclically (closed) or not (open).
    """

    shape: Tuple[int, int]
    h: float
    metric_at: Callable[[float, float], Any]
    is_closed_at: Callable[[float, float], bool]
    origin: Tuple[float, float] = (0.0, 0.0)
    stencil_radius: int = 8
    edge_samples: int = 64
    edge_tol: float = 1e-10

    _stencil_cache: Dict[Tuple[int, int], MixedStencil] = field(
        init=False, repr=False
    )
    _reverse_neighbors: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.ny, self.nx = self.shape
        if self.ny <= 0 or self.nx <= 0:
            raise ValueError("shape must be positive")
        if self.h <= 0.0:
            raise ValueError("h must be positive")
        self._stencil_cache = {}
        self._reverse_neighbors = {}
        self._build_stencil_graph()

    @classmethod
    def from_velocity_field(
        cls,
        W_field: np.ndarray,
        h: float,
        *,
        G_field: Optional[np.ndarray] = None,
        origin: Tuple[float, float] = (0.0, 0.0),
        direction: str = "from_source",
        lambda_threshold: float = 0.0,
        stencil_radius: int = 8,
        edge_samples: int = 64,
        edge_tol: float = 1e-10,
    ) -> "MixedFMASR2D":
        """Build a mixed-regime solver from a cell-centered velocity field.

        The wind that drives the cost is ``sign * W`` where ``sign = -1`` for
        ``direction='from_source'`` (the search proceeds from source nodes
        outward, so the travel-time cost uses the reversed wind) and ``+1``
        for ``direction='to_source'``.  Cells with the resulting
        ``lambda = 1 - g(W, W)`` above ``lambda_threshold`` are treated as
        Randers; the rest as conic Zermelo.
        """
        W = np.asarray(W_field, dtype=float)
        if W.ndim != 3 or W.shape[-1] != 2:
            raise ValueError("W_field must have shape (ny, nx, 2)")
        if not np.all(np.isfinite(W)):
            raise ValueError("W_field must contain only finite values")
        if direction not in {"from_source", "to_source"}:
            raise ValueError("direction must be 'from_source' or 'to_source'")
        if G_field is not None:
            G_arr = np.asarray(G_field, dtype=float)
            if G_arr.shape != W.shape[:2] + (2, 2):
                raise ValueError("G_field must have shape (ny, nx, 2, 2)")
            if not np.all(np.isfinite(G_arr)):
                raise ValueError("G_field must contain only finite values")
            for j in range(G_arr.shape[0]):
                for i in range(G_arr.shape[1]):
                    if not np.allclose(G_arr[j, i], G_arr[j, i].T, atol=1e-12):
                        raise ValueError("G_field entries must be symmetric")
                    if np.min(np.linalg.eigvalsh(G_arr[j, i])) <= 0.0:
                        raise ValueError("G_field entries must be SPD")
        else:
            G_arr = None

        ny, nx = W.shape[:2]
        ox, oy = origin
        lambda_threshold = float(lambda_threshold)
        sign = -1.0 if direction == "from_source" else 1.0

        def _cell_indices(x: float, y: float) -> Tuple[int, int]:
            i = int(round((x - ox) / h))
            j = int(round((y - oy) / h))
            i = min(nx - 1, max(0, i))
            j = min(ny - 1, max(0, j))
            return j, i

        def _wind_and_metric(j: int, i: int) -> Tuple[np.ndarray, np.ndarray, float]:
            wind = sign * W[j, i]
            G_local = (
                np.eye(2, dtype=float) if G_arr is None else G_arr[j, i]
            )
            lam = 1.0 - float(wind @ G_local @ wind)
            return wind, G_local, lam

        def metric_at(x: float, y: float) -> Any:
            j, i = _cell_indices(x, y)
            wind, G_local, lam = _wind_and_metric(j, i)
            if lam > lambda_threshold:
                M, beta = randers_from_zermelo(wind, G_local)
                return AsymmetricEuclideanNorm(M=M, beta=beta)
            return StrongWindZermeloNorm(wind, G=G_local)

        def is_closed_at(x: float, y: float) -> bool:
            j, i = _cell_indices(x, y)
            _, _, lam = _wind_and_metric(j, i)
            return lam > lambda_threshold

        return cls(
            shape=(ny, nx),
            h=h,
            metric_at=metric_at,
            is_closed_at=is_closed_at,
            origin=origin,
            stencil_radius=stencil_radius,
            edge_samples=edge_samples,
            edge_tol=edge_tol,
        )

    def _is_inside(self, node: Tuple[int, int]) -> bool:
        j, i = node
        return 0 <= i < self.nx and 0 <= j < self.ny

    def _cell_position(self, j: int, i: int) -> np.ndarray:
        ox, oy = self.origin
        return np.array([ox + i * self.h, oy + j * self.h], dtype=float)

    def _metric_at_cell(self, j: int, i: int) -> Any:
        x, y = self._cell_position(j, i)
        return self.metric_at(float(x), float(y))

    def _is_closed_at_cell(self, j: int, i: int) -> bool:
        x, y = self._cell_position(j, i)
        return bool(self.is_closed_at(float(x), float(y)))

    def stencil_at(self, j: int, i: int) -> MixedStencil:
        key = (j, i)
        stencil = self._stencil_cache.get(key)
        if stencil is None:
            F = self._metric_at_cell(j, i)
            is_closed = self._is_closed_at_cell(j, i)
            if is_closed:
                offsets = build_f_acute_mesh(F)
            else:
                offsets = build_conic_stencil(
                    F, radius=self.stencil_radius, require_acute=True
                )
            stencil = MixedStencil(offsets=offsets, is_closed=is_closed)
            self._stencil_cache[key] = stencil
        return stencil

    def _build_stencil_graph(self) -> None:
        for j in range(self.ny):
            for i in range(self.nx):
                stencil = self.stencil_at(j, i)
                for idx, (dx, dy) in enumerate(stencil.offsets):
                    y_node = (j + dy, i + dx)
                    if self._is_inside(y_node):
                        self._reverse_neighbors.setdefault(y_node, []).append(
                            (j, i, idx)
                        )

    def solve(self, source_mask: np.ndarray) -> np.ndarray:
        return self.solve_source_mask(source_mask)

    def solve_source_mask(self, source_mask: np.ndarray) -> np.ndarray:
        source_mask = np.asarray(source_mask, dtype=bool)
        if source_mask.shape != self.shape:
            raise ValueError("source_mask has wrong shape")
        if not np.any(source_mask):
            raise ValueError("source_mask is empty")

        d = np.full(self.shape, np.inf, dtype=float)
        accepted = np.zeros(self.shape, dtype=bool)
        heap: List[Tuple[float, Tuple[int, int]]] = []
        for j, i in np.argwhere(source_mask):
            d[int(j), int(i)] = 0.0
            heapq.heappush(heap, (0.0, (int(j), int(i))))

        while heap:
            dist, (j, i) = heapq.heappop(heap)
            if accepted[j, i] or dist != d[j, i]:
                continue
            accepted[j, i] = True
            y_pos = self._cell_position(j, i)

            for xj, xi, local_idx in self._reverse_neighbors.get((j, i), []):
                if accepted[xj, xi]:
                    continue
                x_pos = self._cell_position(xj, xi)
                F_x = self._metric_at_cell(xj, xi)
                stencil = self.stencil_at(xj, xi)
                offsets = stencil.offsets
                m = len(offsets)
                is_closed = stencil.is_closed
                best = d[xj, xi]

                direct_cost = float(F_x(y_pos - x_pos))
                if np.isfinite(direct_cost):
                    cand = direct_cost + d[j, i]
                    if cand < best:
                        best = cand

                if is_closed:
                    neighbor_indices = (
                        (local_idx - 1) % m,
                        (local_idx + 1) % m,
                    )
                else:
                    neighbor_indices = (local_idx - 1, local_idx + 1)

                for neighbor_idx in neighbor_indices:
                    if not is_closed and (
                        neighbor_idx < 0 or neighbor_idx >= m
                    ):
                        continue
                    dx, dy = offsets[neighbor_idx]
                    zj = xj + dy
                    zi = xi + dx
                    if not (0 <= zi < self.nx and 0 <= zj < self.ny):
                        continue
                    if not accepted[zj, zi]:
                        continue
                    z_pos = self._cell_position(zj, zi)
                    cand_edge = self._edge_update(
                        F_x, x_pos, z_pos, y_pos, float(d[zj, zi]), float(d[j, i])
                    )
                    if cand_edge is not None and np.isfinite(cand_edge) and cand_edge < best:
                        best = float(cand_edge)

                if best + 1e-15 < d[xj, xi]:
                    d[xj, xi] = best
                    heapq.heappush(heap, (best, (xj, xi)))

        return d

    def _edge_update(
        self,
        F_x: Any,
        x_pos: np.ndarray,
        a_pos: np.ndarray,
        b_pos: np.ndarray,
        da: float,
        db: float,
    ) -> Optional[float]:
        edge_update = getattr(F_x, "edge_update", None)
        if not callable(edge_update):
            return None
        try:
            try:
                return edge_update(
                    x_pos,
                    a_pos,
                    b_pos,
                    da,
                    db,
                    samples=self.edge_samples,
                    tol=self.edge_tol,
                )
            except TypeError:
                return edge_update(x_pos, a_pos, b_pos, da, db)
        except NotImplementedError:
            return None


def extract_mixed_fm_asr_path(
    solver: MixedFMASR2D,
    dist: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    *,
    max_steps: int = 10000,
    stop_tol: float = 1e-10,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Continuous Hopf-Lax path extraction for :class:`MixedFMASR2D`.

    Reuses :func:`_trace_continuous_path`, which dispatches between cyclic and
    one-sided edge iteration based on each cell's ``MixedStencil.is_closed``
    flag.
    """
    dist = np.asarray(dist, dtype=float)
    source_mask = np.asarray(source_mask, dtype=bool)
    target_mask = np.asarray(target_mask, dtype=bool)
    if source_mask.shape != dist.shape or target_mask.shape != dist.shape:
        raise ValueError("source_mask and target_mask must match dist shape")

    masked = np.where(target_mask, dist, np.inf)
    flat = int(np.argmin(masked))
    travel_value = float(masked.flat[flat])
    if not np.isfinite(travel_value):
        raise RuntimeError("No finite mixed FM-ASR target travel time")
    tj, ti = (int(v) for v in np.unravel_index(flat, dist.shape))
    chosen_target = (tj, ti)
    target_xy = solver._cell_position(tj, ti)

    path_target_to_source = _trace_continuous_path(
        solver,
        dist,
        target_xy,
        max_steps=max_steps,
        stop_tol=stop_tol,
        source_mask=source_mask,
    )
    if path_target_to_source.shape[0] < 2:
        raise RuntimeError(
            "Mixed FM-ASR path extraction produced fewer than two points"
        )
    return path_target_to_source[::-1], travel_value, chosen_target
