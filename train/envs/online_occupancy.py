"""Re-export occupancy helpers from the ROS package (single implementation)."""

from __future__ import annotations

import sys
from pathlib import Path

_DRONE_PKG = Path(__file__).resolve().parents[2] / "src" / "drone_robot"
_pkg = str(_DRONE_PKG)
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)

from drone_robot.online_occupancy import (  # noqa: E402
    GRID_RES,
    INFLATE_M,
    astar,
    carve_rays,
    lookahead_on_path,
    make_grid,
    world_to_index,
)

__all__ = [
    "GRID_RES",
    "INFLATE_M",
    "astar",
    "lookahead_on_path",
    "world_to_index",
    "make_grid",
    "carve_rays",
]
