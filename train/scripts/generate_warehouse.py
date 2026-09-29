#!/usr/bin/env python3
"""Procedural box warehouse: URDF (PyBullet) + SDF world (Gazebo) + occupancy grid.

    python3 train/scripts/generate_warehouse.py --seed 0 --name warehouse
    python3 train/scripts/generate_warehouse.py --seed 101 --name eval_1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "src" / "drone_robot" / "models"
WORLDS = REPO / "src" / "drone_robot" / "worlds"
CONFIG = REPO / "src" / "drone_robot" / "config"

# World frame (m). Spawn/goal sit in the center aisle.
XMIN, XMAX = -8.0, 30.0
YMIN, YMAX = -12.0, 12.0
WALL_T = 0.40
WALL_H = 3.0
RACK_SX, RACK_SY, RACK_SZ = 1.60, 0.80, 2.50
AISLE_YS = (-6.0, 0.0, 6.0)
AISLE_HALF = 1.20
GRID_RES = 0.25
INFLATE = 0.35
SPAWN = np.array([-5.0, 0.0], dtype=np.float64)
GOAL = np.array([26.0, 0.0], dtype=np.float64)


def _box(cx, cy, cz, sx, sy, sz):
    return (float(cx), float(cy), float(cz), float(sx), float(sy), float(sz))


def layout_boxes(rng: np.random.Generator) -> list[tuple[float, ...]]:
    """Outer walls, rack rows, cross-aisle gaps, and seeded dead-end bays."""
    boxes: list[tuple[float, ...]] = []
    cx = 0.5 * (XMIN + XMAX)
    cy = 0.5 * (YMIN + YMAX)
    lx = XMAX - XMIN
    ly = YMAX - YMIN

    boxes.append(_box(cx, YMIN + 0.5 * WALL_T, 0.5 * WALL_H, lx, WALL_T, WALL_H))
    boxes.append(_box(cx, YMAX - 0.5 * WALL_T, 0.5 * WALL_H, lx, WALL_T, WALL_H))
    boxes.append(_box(XMIN + 0.5 * WALL_T, cy, 0.5 * WALL_H, WALL_T, ly, WALL_H))
    boxes.append(_box(XMAX - 0.5 * WALL_T, cy, 0.5 * WALL_H, WALL_T, ly, WALL_H))

    rack_rows_y = (-3.0, 3.0)
    xs = np.arange(-2.0, 24.5, 2.0)
    # Two full-width cross-aisles so A* can change lanes.
    gap_a = float(rng.choice([6.0, 8.0, 10.0]))
    gap_b = float(rng.choice([16.0, 18.0, 20.0]))
    gaps = {gap_a, gap_b}

    for y in rack_rows_y:
        for x in xs:
            if any(abs(float(x) - g) < 1.6 for g in gaps):
                continue
            jitter = float(rng.uniform(-0.15, 0.15))
            boxes.append(
                _box(float(x) + jitter, y, 0.5 * RACK_SZ, RACK_SX, RACK_SY, RACK_SZ)
            )

    # Close the east end of a side aisle (dead-end, must backtrack to a gap).
    close_north = bool(rng.random() < 0.5)
    y_close = 6.0 if close_north else -6.0
    boxes.append(_box(24.5, y_close, 0.5 * WALL_H, 0.4, 2.2, WALL_H))

    # U-shaped pocket off a side aisle: open toward the aisle, closed otherwise.
    pocket_x = float(rng.choice([4.0, 12.0, 14.0]))
    pocket_side = 1.0 if close_north else -1.0
    py = pocket_side * 9.0
    boxes.append(_box(pocket_x - 1.4, py, 0.5 * WALL_H, 0.4, 2.4, WALL_H))
    boxes.append(_box(pocket_x + 1.4, py, 0.5 * WALL_H, 0.4, 2.4, WALL_H))
    boxes.append(_box(pocket_x, py + pocket_side * 1.3, 0.5 * WALL_H, 3.2, 0.4, WALL_H))

    # Extra scattered racks in the outer bays (not the center aisle).
    for _ in range(int(rng.integers(6, 11))):
        x = float(rng.uniform(0.0, 22.0))
        y = float(rng.choice([-9.0, 9.0])) + float(rng.uniform(-0.4, 0.4))
        if abs(x - pocket_x) < 3.0 and abs(y - py) < 3.0:
            continue
        boxes.append(_box(x, y, 0.5 * RACK_SZ, RACK_SX, RACK_SY, RACK_SZ))

    return boxes


def occupancy_grid(boxes):
    xs = np.arange(XMIN, XMAX + 1e-9, GRID_RES)
    ys = np.arange(YMIN, YMAX + 1e-9, GRID_RES)
    nx, ny = xs.size, ys.size
    occ = np.zeros((ny, nx), dtype=np.uint8)
    gx, gy = np.meshgrid(xs, ys)
    pad = INFLATE
    for cx, cy, _cz, sx, sy, _sz in boxes:
        hx, hy = 0.5 * sx + pad, 0.5 * sy + pad
        mask = (np.abs(gx - cx) <= hx) & (np.abs(gy - cy) <= hy)
        occ[mask] = 1
    return xs, ys, occ


def _urdf(boxes, robot_name: str) -> str:
    parts = [
        '<?xml version="1.0"?>',
        f'<robot name="{robot_name}">',
        '  <link name="base_link">',
        "    <inertial><mass value=\"0.01\"/><inertia ixx=\"1e-6\" ixy=\"0\" ixz=\"0\" iyy=\"1e-6\" iyz=\"0\" izz=\"1e-6\"/></inertial>",
        "  </link>",
    ]
    for i, (cx, cy, cz, sx, sy, sz) in enumerate(boxes):
        link = f"box_{i}"
        parts.append(f'  <link name="{link}">')
        parts.append("    <inertial>")
        parts.append(f"      <origin xyz=\"{cx:.4f} {cy:.4f} {cz:.4f}\" rpy=\"0 0 0\"/>")
        parts.append("      <mass value=\"0.01\"/>")
        parts.append("      <inertia ixx=\"1e-4\" ixy=\"0\" ixz=\"0\" iyy=\"1e-4\" iyz=\"0\" izz=\"1e-4\"/>")
        parts.append("    </inertial>")
        parts.append("    <visual>")
        parts.append(f"      <origin xyz=\"{cx:.4f} {cy:.4f} {cz:.4f}\" rpy=\"0 0 0\"/>")
        parts.append("      <geometry>")
        parts.append(f"        <box size=\"{sx:.4f} {sy:.4f} {sz:.4f}\"/>")
        parts.append("      </geometry>")
        parts.append("      <material name=\"rack\"><color rgba=\"0.45 0.42 0.38 1\"/></material>")
        parts.append("    </visual>")
        parts.append("    <collision>")
        parts.append(f"      <origin xyz=\"{cx:.4f} {cy:.4f} {cz:.4f}\" rpy=\"0 0 0\"/>")
        parts.append("      <geometry>")
        parts.append(f"        <box size=\"{sx:.4f} {sy:.4f} {sz:.4f}\"/>")
        parts.append("      </geometry>")
        parts.append("    </collision>")
        parts.append("  </link>")
        parts.append(f'  <joint name="j_{i}" type="fixed">')
        parts.append("    <parent link=\"base_link\"/>")
        parts.append(f"    <child link=\"{link}\"/>")
        parts.append("  </joint>")
    parts.append("</robot>")
    return "\n".join(parts) + "\n"


def _goal_pole_sdf(model_name: str, x: float, y: float, r: float, g: float, b: float) -> str:
    """Visual-only pole (no collision) so LiDAR and A* occupancy stay unchanged."""
    return f"""    <model name="{model_name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} 0 0 0 0</pose>
      <link name="link">
        <visual name="mast">
          <pose>0 0 1.5 0 0 0</pose>
          <geometry><cylinder><radius>0.12</radius><length>3.0</length></cylinder></geometry>
          <material>
            <ambient>{r} {g} {b} 1</ambient>
            <diffuse>{r} {g} {b} 1</diffuse>
            <emissive>{r * 0.4} {g * 0.4} {b * 0.4} 1</emissive>
          </material>
        </visual>
        <visual name="ball">
          <pose>0 0 3.15 0 0 0</pose>
          <geometry><sphere><radius>0.28</radius></sphere></geometry>
          <material>
            <ambient>{r} {g} {b} 1</ambient>
            <diffuse>{r} {g} {b} 1</diffuse>
            <emissive>{r * 0.5} {g * 0.5} {b * 0.5} 1</emissive>
          </material>
        </visual>
      </link>
    </model>
"""


def _sdf_world(name: str, urdf_abs: Path) -> str:
    uri = urdf_abs.as_posix()
    west = _goal_pole_sdf("goal_west", float(SPAWN[0]), float(SPAWN[1]), 0.15, 0.85, 0.25)
    east = _goal_pole_sdf("goal_east", float(GOAL[0]), float(GOAL[1]), 0.15, 0.35, 0.95)
    return f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="{name}">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>0</real_time_factor>
      <real_time_update_rate>0</real_time_update_rate>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <gravity>0 0 -9.8</gravity>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>80 40</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>80 40</size></plane></geometry>
          <material>
            <ambient>0.75 0.75 0.72 1</ambient>
            <diffuse>0.75 0.75 0.72 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    <include>
      <uri>file://{uri}</uri>
      <name>warehouse_layout</name>
      <pose>0 0 0 0 0 0</pose>
    </include>
{west}{east}  </world>
</sdf>
"""


def write_artifacts(name: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    boxes = layout_boxes(rng)
    xs, ys, occ = occupancy_grid(boxes)

    MODELS.mkdir(parents=True, exist_ok=True)
    WORLDS.mkdir(parents=True, exist_ok=True)
    CONFIG.mkdir(parents=True, exist_ok=True)

    urdf_path = MODELS / f"{name}.urdf"
    sdf_path = WORLDS / f"{name}.sdf"
    npz_path = CONFIG / f"{name}_grid.npz"

    urdf_path.write_text(_urdf(boxes, name))
    sdf_path.write_text(_sdf_world(name, urdf_path.resolve()))
    np.savez(
        npz_path,
        occupancy=occ,
        xs=xs,
        ys=ys,
        resolution=np.float64(GRID_RES),
        inflate=np.float64(INFLATE),
        xmin=np.float64(XMIN),
        xmax=np.float64(XMAX),
        ymin=np.float64(YMIN),
        ymax=np.float64(YMAX),
        spawn=SPAWN,
        goal=GOAL,
        seed=np.int64(seed),
        name=np.asarray(name),
    )
    n_occ = int(occ.sum())
    print(
        f"wrote {urdf_path.name}, {sdf_path.name}, {npz_path.name} "
        f"(seed={seed}, boxes={len(boxes)}, occupied={n_occ}/{occ.size})"
    )
    try:
        import sys

        sys.path.insert(0, str(REPO / "src" / "drone_robot"))
        from drone_robot.grid_astar import astar

        path = astar(occ, xs, ys, tuple(SPAWN), tuple(GOAL))
        print(f"  A* spawn->goal waypoints: {len(path)}")
        if len(path) < 2:
            raise RuntimeError(f"no A* path for seed={seed} name={name}")
    except ImportError:
        pass
    return {"urdf": urdf_path, "sdf": sdf_path, "npz": npz_path, "occ": occ, "xs": xs, "ys": ys}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default="warehouse")
    p.add_argument(
        "--eval-suite",
        action="store_true",
        help="also write eval_1..eval_5 with seeds 101..105",
    )
    return p.parse_args()


def main():
    args = parse_args()
    write_artifacts(args.name, args.seed)
    if args.eval_suite:
        for i, seed in enumerate(range(101, 106), start=1):
            write_artifacts(f"eval_{i}", seed)


if __name__ == "__main__":
    main()
