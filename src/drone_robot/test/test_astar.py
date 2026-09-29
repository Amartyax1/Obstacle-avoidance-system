"""A* finds a path through the generated warehouse occupancy grid."""

from pathlib import Path

import numpy as np

from drone_robot.grid_astar import astar, lookahead_on_path, world_to_index

def grid_path() -> Path:
    here = Path(__file__).resolve()
    local = here.parents[1] / "config" / "warehouse_grid.npz"
    if local.is_file():
        return local
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory("drone_robot")) / "config" / "warehouse_grid.npz"


def test_spawn_to_goal_path_exists():
    data = np.load(grid_path(), allow_pickle=True)
    path = astar(
        data["occupancy"],
        data["xs"],
        data["ys"],
        tuple(np.asarray(data["spawn"], dtype=float)),
        tuple(np.asarray(data["goal"], dtype=float)),
    )
    assert len(path) > 10
    assert np.hypot(path[-1][0] - float(data["goal"][0]), path[-1][1] - float(data["goal"][1])) < 1.0


def test_lookahead_is_ahead_of_start():
    data = np.load(grid_path(), allow_pickle=True)
    spawn = tuple(np.asarray(data["spawn"], dtype=float))
    goal = tuple(np.asarray(data["goal"], dtype=float))
    path = astar(data["occupancy"], data["xs"], data["ys"], spawn, goal)
    look = lookahead_on_path(path, spawn, distance=2.0)
    assert look is not None
    d_look = np.hypot(look[0] - spawn[0], look[1] - spawn[1])
    d_goal = np.hypot(goal[0] - spawn[0], goal[1] - spawn[1])
    assert d_look < d_goal
    assert d_look > 1.0


def test_blocking_current_cell_still_has_path():
    data = np.load(grid_path(), allow_pickle=True)
    spawn = tuple(np.asarray(data["spawn"], dtype=float))
    goal = tuple(np.asarray(data["goal"], dtype=float))
    ix, iy = world_to_index(spawn[0], spawn[1], data["xs"], data["ys"])
    path = astar(
        data["occupancy"],
        data["xs"],
        data["ys"],
        spawn,
        goal,
        extra_blocked=(ix, iy),
    )
    assert len(path) > 2
