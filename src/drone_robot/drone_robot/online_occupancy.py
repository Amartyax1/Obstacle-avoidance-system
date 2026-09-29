"""Optimistic live occupancy: unknown is free; carve lidar rays; reuse A*."""

from __future__ import annotations

import math

import numpy as np

from drone_robot.grid_astar import astar, lookahead_on_path, world_to_index

GRID_RES = 0.25
INFLATE_M = 0.25

__all__ = [
    "GRID_RES",
    "INFLATE_M",
    "astar",
    "lookahead_on_path",
    "world_to_index",
    "make_grid",
    "carve_rays",
]


def make_grid(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    resolution: float = GRID_RES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zero occupancy (unknown=free) with warehouse-style xs/ys axes."""
    xs = np.arange(xmin, xmax + 1e-9, resolution)
    ys = np.arange(ymin, ymax + 1e-9, resolution)
    occ = np.zeros((ys.size, xs.size), dtype=np.uint8)
    return occ, xs, ys


def _cells_on_segment(
    xs: np.ndarray,
    ys: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> list[tuple[int, int]]:
    dist = math.hypot(x1 - x0, y1 - y0)
    step = 0.5 * float(xs[1] - xs[0]) if xs.size > 1 else 0.5 * GRID_RES
    n = max(int(math.ceil(dist / max(step, 1e-6))) + 1, 2)
    cells: list[tuple[int, int]] = []
    last: tuple[int, int] | None = None
    for i in range(n):
        t = i / (n - 1)
        ij = world_to_index(x0 + t * (x1 - x0), y0 + t * (y1 - y0), xs, ys)
        if ij != last:
            cells.append(ij)
            last = ij
    return cells


def _inflate_hits(
    occ: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    hits: list[tuple[int, int]],
    inflate_m: float,
) -> None:
    if not hits or inflate_m <= 0.0:
        return
    ny, nx = occ.shape
    res_x = float(xs[1] - xs[0]) if xs.size > 1 else GRID_RES
    res_y = float(ys[1] - ys[0]) if ys.size > 1 else GRID_RES
    rad_x = int(math.ceil(inflate_m / max(res_x, 1e-6)))
    rad_y = int(math.ceil(inflate_m / max(res_y, 1e-6)))
    for hx, hy in hits:
        for dy in range(-rad_y, rad_y + 1):
            for dx in range(-rad_x, rad_x + 1):
                if math.hypot(dx * res_x, dy * res_y) > inflate_m + 1e-6:
                    continue
                jx, jy = hx + dx, hy + dy
                if 0 <= jx < nx and 0 <= jy < ny:
                    occ[jy, jx] = 1


def carve_rays(
    occ: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    origin_xy: tuple[float, float],
    ranges: np.ndarray,
    headings: np.ndarray,
    max_range: float,
    inflate_m: float = INFLATE_M,
) -> None:
    """Walk each beam: free along the ray; occupied+inflate at hits only.

    occupancy: 0 = free/unknown, 1 = occupied. Max-range misses do not occupy.
    """
    ny, nx = occ.shape
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    oix, oiy = world_to_index(ox, oy, xs, ys)
    ranges = np.asarray(ranges, dtype=np.float64).reshape(-1)
    headings = np.asarray(headings, dtype=np.float64).reshape(-1)
    hits: list[tuple[int, int]] = []
    for rng, heading in zip(ranges, headings):
        rng = float(np.clip(rng, 0.0, max_range))
        is_hit = rng < (max_range - 1e-4)
        dist = rng if is_hit else float(max_range)
        ex = ox + dist * math.cos(heading)
        ey = oy + dist * math.sin(heading)
        cells = _cells_on_segment(xs, ys, ox, oy, ex, ey)
        if not cells:
            continue
        free_cells = cells[:-1] if is_hit else cells
        for ix, iy in free_cells:
            if 0 <= ix < nx and 0 <= iy < ny:
                occ[iy, ix] = 0
        if not is_hit:
            continue
        ix, iy = cells[-1]
        if not (0 <= ix < nx and 0 <= iy < ny):
            continue
        if (ix, iy) == (oix, oiy):
            continue
        occ[iy, ix] = 1
        hits.append((ix, iy))
    _inflate_hits(occ, xs, ys, hits, inflate_m)
