"""Carve + A* unit tests (no PyBullet)."""

from __future__ import annotations

import numpy as np

from drone_robot.online_occupancy import (
    astar,
    carve_rays,
    lookahead_on_path,
    make_grid,
    world_to_index,
)


def test_carve_hit_frees_beam_and_inflates_endpoint():
    occ, xs, ys = make_grid(0.0, 5.0, -1.0, 1.0)
    origin = (0.1, 0.0)
    carve_rays(
        occ,
        xs,
        ys,
        origin,
        ranges=np.array([2.0]),
        headings=np.array([0.0]),
        max_range=5.0,
        inflate_m=0.25,
    )
    hx, hy = world_to_index(2.0, 0.0, xs, ys)
    assert occ[hy, hx] == 1
    inflated = False
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        jx, jy = hx + dx, hy + dy
        if 0 <= jx < occ.shape[1] and 0 <= jy < occ.shape[0] and occ[jy, jx] == 1:
            inflated = True
    assert inflated
    mx, my = world_to_index(1.0, 0.0, xs, ys)
    assert occ[my, mx] == 0


def test_astar_empty_grid_is_straight():
    occ, xs, ys = make_grid(0.0, 10.0, 0.0, 10.0)
    start, goal = (1.0, 5.0), (8.0, 5.0)
    path = astar(occ, xs, ys, start, goal)
    assert len(path) > 2
    ys_path = np.array([p[1] for p in path])
    assert float(np.max(np.abs(ys_path - 5.0))) < 0.3
    look = lookahead_on_path(path, start, distance=2.0)
    assert look is not None
    assert abs(look[1] - 5.0) < 0.3
    assert look[0] > start[0]


def test_astar_goes_around_or_empty_after_blocking_carve():
    occ, xs, ys = make_grid(0.0, 10.0, 0.0, 10.0)
    start, goal = (1.0, 5.0), (9.0, 5.0)
    open_path = astar(occ, xs, ys, start, goal)
    open_len = len(open_path)
    assert open_len > 2
    for y in np.arange(0.1, 9.9, 0.2):
        carve_rays(
            occ,
            xs,
            ys,
            origin_xy=(0.2, float(y)),
            ranges=np.array([5.0]),
            headings=np.array([0.0]),
            max_range=10.0,
            inflate_m=0.25,
        )
    blocked = astar(occ, xs, ys, start, goal)
    if blocked:
        ys_path = np.array([p[1] for p in blocked])
        detour = float(np.max(np.abs(ys_path - 5.0)))
        assert detour > 0.4 or len(blocked) > open_len
    else:
        assert blocked == []
