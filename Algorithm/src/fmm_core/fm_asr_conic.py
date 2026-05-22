from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fm_asr import golden_section_minimize

ArrayLike = Sequence[float]
Vec = np.ndarray


def _wrap_angle(theta: float) -> float:
    return (theta + math.pi) % (2.0 * math.pi) - math.pi


def _is_primitive(dx: int, dy: int) -> bool:
    return math.gcd(abs(dx), abs(dy)) == 1


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

    def acute(self, u: ArrayLike, v: ArrayLike) -> bool:
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        if not self.is_admissible(u) or not self.is_admissible(v):
            return False
        fu = float(self(u))
        fv = float(self(v))
        if not np.isfinite(fu) or not np.isfinite(fv):
            return False
        for delta in np.concatenate(([0.0], np.logspace(-10, 6, 18))):
            uv = u + delta * v
            vu = v + delta * u
            if not self.is_admissible(uv) or not self.is_admissible(vu):
                return False
            if float(self(uv)) + 1e-12 < fu:
                return False
            if float(self(vu)) + 1e-12 < fv:
                return False
        return True

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
        x = np.asarray(x, dtype=float)
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        samples = max(int(samples), 2)

        def direction(t: float) -> np.ndarray:
            return (1.0 - t) * a + t * b - x

        def phi(t: float) -> float:
            cost = float(self(direction(t)))
            if not np.isfinite(cost):
                return float("inf")
            return cost + (1.0 - t) * float(da) + t * float(db)

        ts = np.linspace(0.0, 1.0, samples + 1)
        valid = np.array([self.is_admissible(direction(float(t))) for t in ts])
        intervals: List[Tuple[float, float]] = []
        start: Optional[float] = None
        last_valid = False

        for idx, ok in enumerate(valid):
            t = float(ts[idx])
            if ok and not last_valid:
                start = t
            if last_valid and not ok and start is not None:
                intervals.append((start, float(ts[idx - 1])))
                start = None
            last_valid = bool(ok)
        if last_valid and start is not None:
            intervals.append((start, 1.0))

        if not intervals:
            return None

        best_t: Optional[float] = None
        best_val = float("inf")
        for left, right in intervals:
            if right < left:
                continue
            candidates = [left, right]
            if right - left > tol:
                t_mid, val_mid = golden_section_minimize(phi, left, right, tol=tol)
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
        q = (1.0 - best_t) * a + best_t * b
        return best_t, q, float(best_val)


@dataclass
class ConicStencil:
    offsets: List[Tuple[int, int]]

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return [(idx, idx + 1) for idx in range(len(self.offsets) - 1)]


def build_conic_stencil(
    F: StrongWindZermeloNorm,
    radius: int = 8,
    *,
    require_acute: bool = True,
) -> List[Tuple[int, int]]:
    """Build a sorted admissible integer stencil for a conic norm."""
    radius = int(radius)
    if radius < 1:
        raise ValueError("radius must be >= 1")

    offsets: List[Tuple[int, int]] = []
    center_angle = math.atan2(float(F.W[1]), float(F.W[0]))
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            if not _is_primitive(dx, dy):
                continue
            u = np.array([dx, dy], dtype=float)
            if F.is_admissible(u):
                offsets.append((dx, dy))

    if len(offsets) < 2:
        raise RuntimeError("conic stencil has fewer than two admissible offsets")

    if F.lambda_ > F.eps:
        offsets.sort(key=lambda item: math.atan2(item[1], item[0]))
    else:
        offsets.sort(
            key=lambda item: _wrap_angle(math.atan2(item[1], item[0]) - center_angle)
        )

    if require_acute and len(offsets) > 2:
        acute_offsets = [offsets[0]]
        for candidate in offsets[1:]:
            prev = np.asarray(acute_offsets[-1], dtype=float)
            cur = np.asarray(candidate, dtype=float)
            if F.acute(prev, cur):
                acute_offsets.append(candidate)
        if len(acute_offsets) >= 2:
            offsets = acute_offsets

    return offsets


def dispersion_tensor_field_from_velocity(
    W_field: np.ndarray,
    *,
    molecular_diffusion: float = 0.0,
    alpha_l: float = 0.0,
    alpha_t: float = 0.0,
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


def extract_conic_fm_asr_path(
    solver: ConicFMASR2D,
    dist: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """Extract a simple grid-node path from a conic FM-ASR distance map."""
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
    current = tuple(int(v) for v in np.unravel_index(flat, dist.shape))
    chosen_target = current
    nodes = [current]

    for _ in range(dist.size * 4):
        if source_mask[current]:
            break
        j, i = current
        best = current
        best_value = float(dist[current])
        for dx, dy in solver.stencil_at(j, i).offsets:
            nj = j + dy
            ni = i + dx
            if not (0 <= ni < solver.nx and 0 <= nj < solver.ny):
                continue
            value = float(dist[nj, ni])
            if np.isfinite(value) and value < best_value:
                best = (nj, ni)
                best_value = value
        if best == current:
            break
        current = best
        nodes.append(current)

    points = np.array([solver._cell_position(j, i) for j, i in reversed(nodes)])
    return points, travel_value, chosen_target
