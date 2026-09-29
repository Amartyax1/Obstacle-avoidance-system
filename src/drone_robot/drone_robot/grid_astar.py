"""8-connected A* on a 2D occupancy grid (world xy, row-major occupancy)."""

from __future__ import annotations

import heapq
import math

import numpy as np

NEIGHBORS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1, 1, math.sqrt(2.0)),
    (1, -1, math.sqrt(2.0)),
    (1, 1, math.sqrt(2.0)),
)


def world_to_index(x: float, y: float, xs: np.ndarray, ys: np.ndarray) -> tuple[int, int]:
    ix = int(np.clip(np.searchsorted(xs, x) - 1, 0, xs.size - 1))
    iy = int(np.clip(np.searchsorted(ys, y) - 1, 0, ys.size - 1))
    if ix + 1 < xs.size and abs(xs[ix + 1] - x) < abs(xs[ix] - x):
        ix += 1
    if iy + 1 < ys.size and abs(ys[iy + 1] - y) < abs(ys[iy] - y):
        iy += 1
    return ix, iy


def index_to_world(ix: int, iy: int, xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    return float(xs[ix]), float(ys[iy])


def nearest_free(ix: int, iy: int, occ: np.ndarray) -> tuple[int, int] | None:
    ny, nx = occ.shape
    if 0 <= ix < nx and 0 <= iy < ny and occ[iy, ix] == 0:
        return ix, iy
    for r in range(1, 24):
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                jx, jy = ix + dx, iy + dy
                if 0 <= jx < nx and 0 <= jy < ny and occ[jy, jx] == 0:
                    return jx, jy
    return None


def astar(
    occ: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    extra_blocked: tuple[int, int] | None = None,
) -> list[tuple[float, float]]:
    """Return world-frame waypoints from start to goal, or [] if none."""
    ny, nx = occ.shape
    blocked = occ.copy()
    if extra_blocked is not None:
        bx, by = extra_blocked
        if 0 <= bx < nx and 0 <= by < ny:
            blocked[by, bx] = 1

    s = nearest_free(*world_to_index(start_xy[0], start_xy[1], xs, ys), blocked)
    g = nearest_free(*world_to_index(goal_xy[0], goal_xy[1], xs, ys), blocked)
    if s is None or g is None:
        return []
    sx, sy = s
    gx, gy = g
    if (sx, sy) == (gx, gy):
        return [index_to_world(sx, sy, xs, ys)]

    def h(ix, iy):
        return math.hypot(ix - gx, iy - gy)

    open_h = [(h(sx, sy), 0.0, sx, sy)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    gscore = {(sx, sy): 0.0}
    closed: set[tuple[int, int]] = set()

    while open_h:
        _, cost, ix, iy = heapq.heappop(open_h)
        if (ix, iy) in closed:
            continue
        if (ix, iy) == (gx, gy):
            path_ij = [(ix, iy)]
            while (ix, iy) in came:
                ix, iy = came[(ix, iy)]
                path_ij.append((ix, iy))
            path_ij.reverse()
            return [index_to_world(px, py, xs, ys) for px, py in path_ij]
        closed.add((ix, iy))
        for dx, dy, step in NEIGHBORS:
            jx, jy = ix + dx, iy + dy
            if not (0 <= jx < nx and 0 <= jy < ny):
                continue
            if blocked[jy, jx]:
                continue
            # No diagonal corner-cutting through occupied cells.
            if dx != 0 and dy != 0 and (blocked[iy, jx] or blocked[jy, ix]):
                continue
            ng = cost + step
            key = (jx, jy)
            if ng + 1e-9 < gscore.get(key, math.inf):
                gscore[key] = ng
                came[key] = (ix, iy)
                heapq.heappush(open_h, (ng + h(jx, jy), ng, jx, jy))
    return []


def lookahead_on_path(
    path: list[tuple[float, float]],
    pose_xy: tuple[float, float],
    distance: float = 2.0,
) -> tuple[float, float] | None:
    if not path:
        return None
    px, py = pose_xy
    nearest = min(range(len(path)), key=lambda i: math.hypot(path[i][0] - px, path[i][1] - py))
    remaining = 0.0
    x0, y0 = path[nearest]
    if nearest == 0:
        remaining = 0.0
    for i in range(nearest, len(path) - 1):
        x1, y1 = path[i + 1]
        seg = math.hypot(x1 - x0, y1 - y0)
        if remaining + seg >= distance:
            t = (distance - remaining) / max(seg, 1e-6)
            return x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        remaining += seg
        x0, y0 = x1, y1
    return path[-1]
