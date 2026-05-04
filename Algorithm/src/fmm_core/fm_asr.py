from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

ArrayLike = Sequence[float]
Vec = np.ndarray


def golden_section_minimize(
    f: Callable[[float], float],
    a: float = 0.0,
    b: float = 1.0,
    tol: float = 1e-12,
    max_iter: int = 80,
) -> Tuple[float, float]:
    """Minimize a convex 1D function on [a, b] with golden-section search."""
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(max_iter):
        if abs(b - a) <= tol:
            break
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = f(d)
    t = c if fc <= fd else d
    return t, min(fc, fd)


class FinslerNorm:
    """Base class for a Finsler/asymmetric norm.

    Subclasses may implement any of the following optional hooks:
    - grad(u): gradient for smooth norms
    - acute(u, v): exact F-acute predicate
    - edge_update(a, b, da, db, x): exact Hopf-Lax update on a segment
    """

    def __call__(self, u: ArrayLike) -> float:
        raise NotImplementedError

    def grad(self, u: ArrayLike) -> Vec:
        raise NotImplementedError

    def acute(self, u: ArrayLike, v: ArrayLike) -> bool:
        raise NotImplementedError

    def edge_update(
        self,
        x: ArrayLike,
        a: ArrayLike,
        b: ArrayLike,
        da: float,
        db: float,
    ) -> Optional[float]:
        return None


@dataclass
class AsymmetricEuclideanNorm(FinslerNorm):
    """F(u) = sqrt(u^T M u) - beta^T u.

    This is a common asymmetric norm used in the paper's numerical tests.
    Validity requires beta^T M^{-1} beta < 1.
    """

    M: np.ndarray
    beta: np.ndarray

    def __post_init__(self) -> None:
        self.M = np.asarray(self.M, dtype=float)
        self.beta = np.asarray(self.beta, dtype=float)
        if self.M.shape != (2, 2):
            raise ValueError("M must be 2x2")
        if self.beta.shape != (2,):
            raise ValueError("beta must be a 2-vector")
        if not np.allclose(self.M, self.M.T, atol=1e-12):
            raise ValueError("M must be symmetric")
        eigs = np.linalg.eigvalsh(self.M)
        if np.min(eigs) <= 0:
            raise ValueError("M must be SPD")
        invM = np.linalg.inv(self.M)
        if float(self.beta.T @ invM @ self.beta) >= 1.0:
            raise ValueError("Need beta^T M^{-1} beta < 1")

    def __call__(self, u: ArrayLike) -> float:
        u = np.asarray(u, dtype=float)
        q = float(u.T @ self.M @ u)
        if q <= 0.0:
            return 0.0
        return math.sqrt(q) - float(self.beta @ u)

    def grad(self, u: ArrayLike) -> Vec:
        u = np.asarray(u, dtype=float)
        q = float(u.T @ self.M @ u)
        if q <= 0.0:
            raise ValueError("Gradient undefined at the origin")
        return (self.M @ u) / math.sqrt(q) - self.beta

    def edge_update(
        self,
        x: ArrayLike,
        a: ArrayLike,
        b: ArrayLike,
        da: float,
        db: float,
    ) -> Optional[float]:
        """Closed-form segment minimization for the asymmetric Euclidean norm.

        The objective is
            phi(t) = F((1-t)a + t b - x) + (1-t) da + t db,  t in [0,1].
        It is convex, and stationary points are the real roots of a quadratic.
        """
        x = np.asarray(x, dtype=float)
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        r = a - x
        s = b - a

        A = float(s.T @ self.M @ s)
        B = float(2.0 * (r.T @ self.M @ s))
        C = float(r.T @ self.M @ r)
        c = float(db - da - self.beta @ s)
        const = float(da - self.beta @ r)

        def phi(t: float) -> float:
            q = A * t * t + B * t + C
            if q < 0.0:
                q = 0.0
            return math.sqrt(q) + c * t + const

        candidates = [0.0, 1.0]

        if A > 1e-18:
            # Solve (2At + B)^2 = 4 c^2 (At^2 + Bt + C)
            p2 = A * (A - c * c)
            p1 = B * (A - c * c)
            p0 = 0.25 * B * B - c * c * C
            if abs(p2) <= 1e-18:
                if abs(p1) > 1e-18:
                    t = -p0 / p1
                    if -1e-12 <= t <= 1.0 + 1e-12:
                        candidates.append(min(1.0, max(0.0, t)))
            else:
                disc = p1 * p1 - 4.0 * p2 * p0
                if disc >= -1e-14:
                    disc = max(0.0, disc)
                    sqrt_disc = math.sqrt(disc)
                    # Numerically stable quadratic formula.
                    q = -0.5 * (p1 + math.copysign(sqrt_disc, p1))
                    if abs(p2) > 1e-18:
                        t1 = q / p2
                        t2 = p0 / q if abs(q) > 1e-18 else t1
                        for t in (t1, t2):
                            if -1e-12 <= t <= 1.0 + 1e-12:
                                candidates.append(min(1.0, max(0.0, t)))
                    elif abs(p1) > 1e-18:
                        t = -p0 / p1
                        if -1e-12 <= t <= 1.0 + 1e-12:
                            candidates.append(min(1.0, max(0.0, t)))
        else:
            # Degenerate segment: objective is affine.
            pass

        best = min(phi(t) for t in candidates)
        return best


def build_f_acute_mesh(F: FinslerNorm, max_iter: int = 10000) -> List[Tuple[int, int]]:
    """Build the unique cyclic boundary vertex list of T(F)."""
    L: List[Tuple[int, int]] = [(1, 0)]
    M: List[Tuple[int, int]] = [(1, 0), (0, -1), (-1, 0), (0, 1)]
    it = 0
    while M:
        it += 1
        if it > max_iter:
            raise RuntimeError("stencil construction did not terminate")
        u = np.asarray(L[-1], dtype=float)
        v = np.asarray(M[-1], dtype=float)
        if is_f_acute(u, v, F):
            L.append(M.pop())
        else:
            w = (int(u[0] + v[0]), int(u[1] + v[1]))
            M.append(w)
    if len(L) > 1 and L[-1] == L[0]:
        L.pop()
    return L


def validate_f_acute_mesh(
    F: FinslerNorm,
    offsets: Sequence[Tuple[int, int]],
    eps: float = 1e-12,
) -> None:
    """Validate the elementary F-acute triangles around a cyclic stencil."""
    cyclic = list(offsets)
    if len(cyclic) < 3:
        raise ValueError("an F-acute mesh needs at least three boundary vertices")
    if len(set(cyclic)) != len(cyclic):
        raise ValueError("stencil boundary vertices must be unique")

    for idx, (u_tuple, v_tuple) in enumerate(zip(cyclic, cyclic[1:] + cyclic[:1])):
        u = np.asarray(u_tuple, dtype=float)
        v = np.asarray(v_tuple, dtype=float)
        if np.allclose(u, 0.0, atol=eps) or np.allclose(v, 0.0, atol=eps):
            raise ValueError("stencil boundary vertices must be non-zero")
        det = int(u_tuple[0] * v_tuple[1] - u_tuple[1] * v_tuple[0])
        if abs(det) != 1:
            raise ValueError(f"stencil edge {idx} is not elementary: det={det}")
        if not is_f_acute(u, v, F, eps=eps):
            raise ValueError(f"stencil edge {idx} is not F-acute")


@dataclass
class LocalStencil:
    offsets: List[Tuple[int, int]]

    @property
    def cyclic_offsets(self) -> List[Tuple[int, int]]:
        return self.offsets + [self.offsets[0]]

    @property
    def edges(self) -> List[Tuple[int, int]]:
        return [(idx, (idx + 1) % len(self.offsets)) for idx in range(len(self.offsets))]


class NonsmoothFallbackWarning(UserWarning):
    pass


def is_f_acute(u: ArrayLike, v: ArrayLike, F: FinslerNorm, eps: float = 1e-12) -> bool:
    """Check F-acuteness.

    Preference order:
    1. exact user-provided acute(u, v)
    2. smooth-gradient criterion when grad() is available
    3. conservative numerical fallback on a logarithmic delta grid

    The fallback avoids crashing on nonsmooth norms, but for exact results on
    polyhedral or otherwise nonsmooth metrics, subclasses should implement acute().
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)

    acute_method = getattr(F, "acute", None)
    if callable(acute_method):
        try:
            return bool(acute_method(u, v))
        except NotImplementedError:
            pass

    try:
        gu = F.grad(u)
        gv = F.grad(v)
        return float(u @ gv) >= -eps and float(v @ gu) >= -eps
    except NotImplementedError:
        pass
    except Exception:
        # Fall through to numerical test.
        pass

    # Numerical fallback: test monotonicity on a finite set of deltas.
    # This is deliberately conservative and intended as a safety net.
    deltas = np.concatenate(
        ([0.0], np.logspace(-12, 6, 24, base=10.0), np.array([1e8, 1e10]))
    )
    fu = float(F(u))
    fv = float(F(v))
    for delta in deltas:
        if float(F(u + delta * v)) + eps < fu:
            return False
        if float(F(v + delta * u)) + eps < fv:
            return False
    return True


@dataclass
class FMASR2D:
    """Fast Marching with Anisotropic Stencil Refinement on a 2D grid.

    The solver supports heterogeneous media by allowing a location-dependent
    metric field. It precomputes or lazily caches the local stencils per grid
    cell, and then builds reverse adjacency from those per-cell stencils.

    Parameters
    ----------
    shape:
        Grid shape as (ny, nx).
    h:
        Grid spacing.
    F:
        Constant Finsler norm. Use this for homogeneous media.
    metric_at:
        Optional callable returning the local Finsler norm at a physical point
        (x, y). If provided, it overrides F and allows spatially varying metrics.
    origin:
        Physical coordinates of grid index (0, 0).
    stencil_strategy:
        Must be "precompute". The paper-style reverse stencil graph is built
        before marching so the solve remains Dijkstra-like.
    """

    shape: Tuple[int, int]
    h: float
    F: Optional[FinslerNorm] = None
    metric_at: Optional[Callable[[float, float], FinslerNorm]] = None
    origin: Tuple[float, float] = (0.0, 0.0)
    stencil_strategy: str = "precompute"

    _stencil_cache: Dict[Tuple[int, int], LocalStencil] = field(init=False, repr=False)
    _reverse_neighbors: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = field(
        init=False, repr=False
    )
    _boundary_nodes: Set[Tuple[int, int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.ny, self.nx = self.shape
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("shape must be positive")
        if self.h <= 0:
            raise ValueError("h must be positive")
        if self.metric_at is None:
            if self.F is None:
                raise ValueError("Provide either a constant F or a metric_at callable")

            def _const_metric_at(_x: float, _y: float) -> FinslerNorm:
                assert self.F is not None
                return self.F

            self.metric_at = _const_metric_at
        self._stencil_cache = {}
        self._reverse_neighbors = {}
        self._boundary_nodes = set()
        if self.stencil_strategy != "precompute":
            raise ValueError("stencil_strategy must be 'precompute'")
        self._build_stencil_graph()

    def _is_inside(self, node: Tuple[int, int]) -> bool:
        j, i = node
        return 0 <= i < self.nx and 0 <= j < self.ny

    def _cell_position(self, j: int, i: int) -> np.ndarray:
        ox, oy = self.origin
        return np.array([ox + i * self.h, oy + j * self.h], dtype=float)

    def _metric_at_cell(self, j: int, i: int) -> FinslerNorm:
        x, y = self._cell_position(j, i)
        return self.metric_at(float(x), float(y))  # type: ignore[misc]

    def stencil_at(self, j: int, i: int) -> LocalStencil:
        """Return the local stencil at grid cell (j, i), using a cache."""
        key = (j, i)
        stencil = self._stencil_cache.get(key)
        if stencil is None:
            stencil = LocalStencil(build_f_acute_mesh(self._metric_at_cell(j, i)))
            self._stencil_cache[key] = stencil
        return stencil

    @property
    def boundary_nodes(self) -> Set[Tuple[int, int]]:
        """Exterior discrete boundary nodes in partial Omega*."""
        return set(self._boundary_nodes)

    def _build_stencil_graph(self) -> None:
        """Precompute the paper-style reverse stencil graph.

        Keys may be interior nodes or exterior nodes in partial Omega*. Values
        are interior nodes whose local stencil contains that key as a vertex.
        """
        for j in range(self.ny):
            for i in range(self.nx):
                stencil = self.stencil_at(j, i)
                for idx, (dx, dy) in enumerate(stencil.offsets):
                    y_node = (j + dy, i + dx)
                    self._reverse_neighbors.setdefault(y_node, []).append((j, i, idx))

        self._boundary_nodes = {
            node for node in self._reverse_neighbors if not self._is_inside(node)
        }

    def solve(self, source_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute the distance map.

        With no ``source_mask``, this is the paper-style escape-time solve with
        the exterior discrete boundary ``partial Omega*`` set to zero. Passing a
        source mask uses the legacy in-domain source mode.
        """
        if source_mask is None:
            return self.solve_escape_time()
        return self.solve_source_mask(source_mask)

    def solve_escape_time(self) -> np.ndarray:
        """Compute escape time using the paper's exterior discrete boundary."""
        ny, nx = self.shape
        d = np.full((ny, nx), np.inf, dtype=float)
        accepted = np.zeros((ny, nx), dtype=bool)
        accepted_boundary: Set[Tuple[int, int]] = set()
        heap: List[Tuple[float, Tuple[int, int]]] = []

        for node in sorted(self._boundary_nodes):
            heapq.heappush(heap, (0.0, node))

        def node_distance(node: Tuple[int, int]) -> float:
            if self._is_inside(node):
                return float(d[node[0], node[1]])
            return 0.0

        def node_accepted(node: Tuple[int, int]) -> bool:
            if self._is_inside(node):
                return bool(accepted[node[0], node[1]])
            return node in accepted_boundary

        while heap:
            dist, y_node = heapq.heappop(heap)
            yj, yi = y_node
            if self._is_inside(y_node):
                if accepted[yj, yi] or dist != d[yj, yi]:
                    continue
                accepted[yj, yi] = True
            else:
                if y_node in accepted_boundary:
                    continue
                accepted_boundary.add(y_node)

            y_pos = self._cell_position(yj, yi)
            dy_value = node_distance(y_node)

            for xj, xi, local_idx in self._reverse_neighbors.get(y_node, []):
                if accepted[xj, xi]:
                    continue

                x_pos = self._cell_position(xj, xi)
                F_x = self._metric_at_cell(xj, xi)
                stencil = self.stencil_at(xj, xi)
                offsets = stencil.offsets
                m = len(offsets)
                best = d[xj, xi]

                cand = float(F_x(y_pos - x_pos)) + dy_value
                if cand < best:
                    best = cand

                for neighbor_idx in ((local_idx - 1) % m, (local_idx + 1) % m):
                    dx, dy = offsets[neighbor_idx]
                    z_node = (xj + dy, xi + dx)
                    if not node_accepted(z_node):
                        continue

                    z_pos = self._cell_position(z_node[0], z_node[1])
                    dz_value = node_distance(z_node)
                    cand_edge = self._edge_update(
                        F_x=F_x,
                        x_pos=x_pos,
                        a_pos=z_pos,
                        b_pos=y_pos,
                        da=dz_value,
                        db=dy_value,
                    )
                    if cand_edge < best:
                        best = cand_edge

                if best + 1e-15 < d[xj, xi]:
                    d[xj, xi] = best
                    heapq.heappush(heap, (best, (xj, xi)))

        return d

    def solve_source_mask(self, source_mask: np.ndarray) -> np.ndarray:
        """Compute the distance map.

        Parameters
        ----------
        source_mask:
            Boolean array of shape (ny, nx) marking zero-cost boundary/source nodes.
        """
        ny, nx = self.shape
        source_mask = np.asarray(source_mask, dtype=bool)
        if source_mask.shape != (ny, nx):
            raise ValueError("source_mask has wrong shape")

        d = np.full((ny, nx), np.inf, dtype=float)
        accepted = np.zeros((ny, nx), dtype=bool)
        heap: List[Tuple[float, Tuple[int, int]]] = []

        for j in range(ny):
            for i in range(nx):
                if source_mask[j, i]:
                    d[j, i] = 0.0
                    heapq.heappush(heap, (0.0, (j, i)))

        while heap:
            dist, (j, i) = heapq.heappop(heap)
            if accepted[j, i] or dist != d[j, i]:
                continue
            accepted[j, i] = True

            y_pos = self._cell_position(j, i)

            # Relax all points x for which the accepted point y is a vertex of the
            # local stencil of x. The reverse adjacency is built from the per-cell
            # stencils, so the solver no longer depends on a single global list.
            for xj, xi, local_idx in self._reverse_neighbors.get((j, i), []):
                if accepted[xj, xi]:
                    continue

                x_pos = self._cell_position(xj, xi)
                F_x = self._metric_at_cell(xj, xi)
                stencil = self.stencil_at(xj, xi)
                offsets = stencil.offsets
                m = len(offsets)
                best = d[xj, xi]

                # Vertex update: direct segment from x to y.
                cand = float(F_x(y_pos - x_pos)) + d[j, i]
                if cand < best:
                    best = cand

                # Edge updates: y may lie on an edge of the stencil boundary.
                for neighbor_idx in ((local_idx - 1) % m, (local_idx + 1) % m):
                    dx, dy = offsets[neighbor_idx]
                    zj = xj + dy
                    zi = xi + dx
                    if not (0 <= zi < nx and 0 <= zj < ny):
                        continue
                    if not accepted[zj, zi]:
                        continue

                    a_pos = self._cell_position(zj, zi)
                    b_pos = y_pos
                    da = d[zj, zi]
                    db = d[j, i]

                    cand_edge = self._edge_update(F_x, x_pos, a_pos, b_pos, da, db)

                    if cand_edge < best:
                        best = cand_edge

                if best + 1e-15 < d[xj, xi]:
                    d[xj, xi] = best
                    heapq.heappush(heap, (best, (xj, xi)))

        return d

    def _edge_update(
        self,
        F_x: FinslerNorm,
        x_pos: np.ndarray,
        a_pos: np.ndarray,
        b_pos: np.ndarray,
        da: float,
        db: float,
    ) -> float:
        cand_edge = None
        if hasattr(F_x, "edge_update"):
            try:
                cand_edge = F_x.edge_update(x_pos, a_pos, b_pos, da, db)
            except NotImplementedError:
                cand_edge = None

        if cand_edge is None:
            def phi(t: float) -> float:
                y_seg = (1.0 - t) * a_pos + t * b_pos
                return float(F_x(y_seg - x_pos)) + (1.0 - t) * da + t * db

            _, cand_edge = golden_section_minimize(phi)

        return float(cand_edge)


if __name__ == "__main__":
    # Homogeneous example.
    M = np.array([[4.0, 0.0], [0.0, 1.0]], dtype=float)
    beta = np.array([0.2, 0.0], dtype=float)
    F = AsymmetricEuclideanNorm(M=M, beta=beta)

    solver = FMASR2D(shape=(51, 51), h=1.0, F=F)
    dist = solver.solve()
    print("stencil size at center:", len(solver.stencil_at(25, 25).offsets))
    print("center distance:", dist[25, 25])
