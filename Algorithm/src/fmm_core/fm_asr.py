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

    @classmethod
    def from_velocity_field(
        cls,
        W_field: np.ndarray,
        h: float,
        origin: Tuple[float, float] = (0.0, 0.0),
        perp_ratio: float = 0.5,
        back_factor: float = 2.0,
        min_speed: float = 1e-6,
        isotropic_only: bool = False,
        direction: str = "from_source",
        stencil_strategy: str = "precompute",
    ) -> "FMASR2D":
        """Build an FM-ASR solver from a cell-centered velocity field.

        ``direction="from_source"`` is the physical source-outward travel-time
        convention. Since the marching update queries the cost from trial nodes
        back to accepted nodes, this mode uses the reversed Randers norm.
        ``direction="to_source"`` keeps the escape/backward orientation.
        """
        c_iso, M_field, beta_field, _ = build_randers_from_W(
            W_field,
            perp_ratio=perp_ratio,
            back_factor=back_factor,
            min_speed=min_speed,
            isotropic_only=isotropic_only,
        )
        del c_iso

        if direction == "from_source":
            solver_beta = -beta_field
        elif direction == "to_source":
            solver_beta = beta_field
        else:
            raise ValueError("direction must be 'from_source' or 'to_source'")

        condition = check_randers_condition(M_field, solver_beta)
        if not np.all(np.isfinite(condition)) or np.nanmax(condition) >= 1.0:
            raise ValueError("velocity-derived Randers field violates beta^T M^{-1} beta < 1")

        ny, nx = M_field.shape[:2]
        ox, oy = origin

        def metric_at(x: float, y: float) -> FinslerNorm:
            i = int(round((x - ox) / h))
            j = int(round((y - oy) / h))
            i = min(nx - 1, max(0, i))
            j = min(ny - 1, max(0, j))
            return AsymmetricEuclideanNorm(M=M_field[j, i], beta=solver_beta[j, i])

        return cls(
            shape=(ny, nx),
            h=h,
            metric_at=metric_at,
            origin=origin,
            stencil_strategy=stencil_strategy,
        )

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

def randers_r_of_theta(M, b, theta):
    M = np.asarray(M, dtype=float)
    b = np.asarray(b, dtype=float)
    d = np.stack([np.cos(theta), np.sin(theta)], axis=0)  # (2, K)
    # compute sqrt(d^T M d) for each theta
    Md = M @ d                         # (2, K)
    quad = np.sum(d * Md, axis=0)      # (K,)
    sym = np.sqrt(np.maximum(quad, 0))

    asym = b @ d                       # (K,)
    denom = sym - asym                 # (K,)

    # denom must be > 0 for a valid Randers indicatrix in that direction
    # mask invalid directions (can happen if b is too strong)
    valid = denom > 1e-12
    r = np.empty_like(denom)
    r[valid] = 1.0 / denom[valid]
    r[~valid] = np.nan
    return r, valid

def plot_randers_indicatrix(M, b, K=720, ax=None, show_vectors=True):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("plot_randers_indicatrix requires matplotlib") from exc

    M = np.asarray(M, dtype=float)
    b = np.asarray(b, dtype=float)
    theta = np.linspace(0, 2*np.pi, K, endpoint=False)
    r, valid = randers_r_of_theta(M, b, theta)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x, y)
    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Randers indicatrix: F(u)=1")

    if show_vectors:
        ax.quiver(0, 0, b[0], b[1], angles="xy", scale_units="xy", scale=1, width=0.003)
        # optionally plot principal directions of M^{-1} as "fast axes" (not required)

    # report invalid angles if any
    frac_invalid = np.mean(~valid)
    if frac_invalid > 0:
        ax.text(0.02, 0.98, f"Invalid directions: {100*frac_invalid:.1f}%",
                transform=ax.transAxes, va="top")

    return ax

def build_randers_from_W(
    W_field,                 # (ny,nx,2) vector field
    perp_ratio=0.5,          # in (0,1]: sideways speed = perp_ratio * |W|
    back_factor=2.0,         # >=1: backward speed = |W| / back_factor
    min_speed=1e-6,          # floor for |W| to avoid divide-by-zero
    isotropic_only=False     # if True: returns pure isotropic metric (b=0, M=I/|W|^2)
):
    """
    Construct Randers metric F(u)=sqrt(u^T M u) - b·u from a given velocity field W(x),
    such that:
      - baseline isotropic speed is c(x)=|W(x)|
      - when isotropic_only=True => M=I/c^2 and b=0 exactly
      - otherwise, we enforce:
            c_eff(+e) = |W|
            c_eff(-e) = |W|/back_factor
            c_eff(perp) = perp_ratio*|W|
        where e = W/|W| and c_eff(d)=1/F(d) for unit direction d.

    Returns:
      c_iso : (ny,nx)  baseline speed c(x)=|W| (floored by min_speed)
      M     : (ny,nx,2,2)
      b     : (ny,nx,2)
      e     : (ny,nx,2) unit direction W/|W| (arbitrary where |W|~0)
    """
    W = np.asarray(W_field, dtype=float)
    if W.ndim != 3 or W.shape[-1] != 2:
        raise ValueError("W_field must have shape (ny,nx,2).")
    if not np.all(np.isfinite(W)):
        raise ValueError("W_field must contain only finite values")
    if min_speed <= 0.0:
        raise ValueError("min_speed must be positive")

    ny, nx, _ = W.shape

    s = np.linalg.norm(W, axis=2)                   # |W|
    c_iso = np.maximum(s, float(min_speed))         # baseline isotropic speed

    # Allocate
    M = np.zeros((ny, nx, 2, 2), dtype=float)
    b = np.zeros((ny, nx, 2), dtype=float)
    e = np.zeros((ny, nx, 2), dtype=float)

    # Validate knobs
    if not (0.0 < perp_ratio <= 1.0):
        raise ValueError("perp_ratio must be in (0,1].")
    if back_factor < 1.0:
        raise ValueError("back_factor must be >= 1.")

    # If you want the isotropic approximation exactly:
    if isotropic_only:
        for j in range(ny):
            for i in range(nx):
                c = c_iso[j, i]
                M[j, i] = np.eye(2) / (c * c)
                b[j, i] = 0.0
                # direction (just for reference)
                if s[j, i] > 0:
                    e[j, i] = W[j, i] / s[j, i]
                else:
                    e[j, i] = np.array([1.0, 0.0])
        return c_iso, M, b, e

    # Otherwise: anisotropic + asymmetric refinement
    for j in range(ny):
        for i in range(nx):
            c = c_iso[j, i]

            # unit direction e = W/|W|
            if s[j, i] > 0:
                ex, ey = W[j, i] / s[j, i]
            else:
                # arbitrary if |W|=0 (metric reverts to isotropic-ish via c_iso floor)
                ex, ey = 1.0, 0.0
            e[j, i] = np.array([ex, ey])

            # perpendicular direction
            px, py = -ey, ex

            # rotation matrix R = [e, e_perp]
            R = np.array([[ex, px],
                          [ey, py]], dtype=float)

            # Target directional speeds
            c_plus  = c                       # max speed along +e: |W|
            c_minus = c / back_factor          # reduced speed along -e
            c_perp  = perp_ratio * c           # reduced sideways speed

            # Choose Randers parameters to match c_plus and c_minus exactly.
            # For unit direction d:
            #   F(d) = sqrt(d^T M d) - b·d
            # Along +e: F(+e) = 1/c_plus
            # Along -e: F(-e) = 1/c_minus
            #
            # If we set (in the local (e, e_perp) frame):
            #   sqrt(d^T M d) gives 1/c_sym_para along ±e
            #   b·d = beta * (e·d) with b parallel to e
            #
            # Then:
            #   1/c_sym_para - beta = 1/c_plus
            #   1/c_sym_para + beta = 1/c_minus
            inv_csym_para = 0.5 * (1.0/c_plus + 1.0/c_minus)
            beta          = 0.5 * (1.0/c_minus - 1.0/c_plus)  # >=0 if c_minus<=c_plus

            c_sym_para = 1.0 / inv_csym_para
            c_sym_perp = c_perp  # keep perpendicular purely symmetric

            # Build symmetric tensor M (slowness^2 tensor)
            D = np.diag([1.0/(c_sym_para**2), 1.0/(c_sym_perp**2)])
            M[j, i] = R @ D @ R.T

            # Build b (covector) aligned with e
            b[j, i] = beta * np.array([ex, ey], dtype=float)

    return c_iso, M, b, e

def check_randers_condition(M0, b):
    M0 = np.asarray(M0, dtype=float)
    b = np.asarray(b, dtype=float)
    if M0.shape[:2] != b.shape[:2] or M0.shape[-2:] != (2, 2) or b.shape[-1] != 2:
        raise ValueError("M0 must have shape (ny,nx,2,2) and b must have shape (ny,nx,2)")
    ny, nx = b.shape[:2]
    vals = np.zeros((ny, nx), float)
    for j in range(ny):
        for i in range(nx):
            Minv = np.linalg.inv(M0[j, i])
            vals[j, i] = np.sqrt(b[j, i] @ (Minv @ b[j, i]))
    return vals  # should be < 1 everywhere

def randers_indicatrix_points(M, b, K=240, eps=1e-12):
    """
    Points u(theta) on indicatrix F(u)=1 for F(u)=sqrt(u^T M u)-b·u.
    u = r(theta) d(theta), r = 1/(sqrt(d^T M d) - b·d)
    """
    M = np.asarray(M, dtype=float)
    b = np.asarray(b, dtype=float)
    theta = np.linspace(0, 2*np.pi, K, endpoint=False)
    d = np.stack([np.cos(theta), np.sin(theta)], axis=0)  # (2,K)
    Md = M @ d
    quad = np.sum(d * Md, axis=0)
    sym = np.sqrt(np.maximum(quad, 0.0))
    asym = b @ d
    denom = sym - asym

    valid = denom > eps
    r = np.full_like(denom, np.nan, dtype=float)
    r[valid] = 1.0 / denom[valid]

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def visualize_indicatrix_field(
    W_field,
    perp_ratio=0.5,
    back_factor=2.0,
    h=1.0,
    stride=10,
    K=180,
    glyph_scale=0.30,          # scales glyph size in plot units (~h)
    show_W_quiver=True,
    quiver_stride=None
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba
    except ImportError as exc:
        raise ImportError("visualize_indicatrix_field requires matplotlib") from exc
    try:
        import cmasher as cm
        cmap = cm.batlowW.reversed()
    except ImportError:
        cmap = "viridis"

    W = np.asarray(W_field, float)
    if W.ndim != 3 or W.shape[-1] != 2:
        raise ValueError("W_field must have shape (ny,nx,2).")
    ny, nx, _ = W.shape

    c_iso, M_field, b_field, _ = build_randers_from_W(
        W, perp_ratio=perp_ratio, back_factor=back_factor
    )
    del c_iso

    if quiver_stride is None:
        quiver_stride = stride

    fig, ax = plt.subplots()

    # Optional: plot W in the background
    if show_W_quiver:
        ys = np.arange(0, ny, quiver_stride)
        xs = np.arange(0, nx, quiver_stride)
        X, Y = np.meshgrid(xs * h, ys * h)
        U = W[ys[:, None], xs[None, :], 0]
        V = W[ys[:, None], xs[None, :], 1]
        v_mag_log = np.log10(np.sqrt(W[:,:,0]**2 + W[:,:,1]**2))
        # ax.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=None, width=0.0025)
        ax.streamplot(X, Y, U, V, color=to_rgba('black', alpha=0.45), linewidth=0.8, arrowsize=0.6)
        im = ax.imshow(v_mag_log, origin='lower', cmap=cmap)
        fig.colorbar(im, ax=ax, label='$\\log_{10} |\\mathbf{v}|$', fraction=0.046, pad=0.04)


    # Indicatrix glyphs
    for j in range(0, ny, stride):
        for i in range(0, nx, stride):
            M = M_field[j, i]
            b = b_field[j, i]
            xi, yi = i * h, j * h  # (x,y) from (j,i)

            gx, gy = randers_indicatrix_points(M, b, K=K)
            ax.plot(xi + glyph_scale * h * gx, yi + glyph_scale * h * gy, color='white', linewidth=0.8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Randers indicatrix field (stride={stride}, perp_ratio={perp_ratio}, back_factor={back_factor})")
    plt.show()
