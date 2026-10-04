#!/usr/bin/env python3
"""Write worlds/explore.sdf: 5x longer west-east flight, same obstacle density.

    python3 train/scripts/generate_explore.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "src" / "drone_robot" / "worlds" / "explore.sdf"

# Original 20x20 m, spawn -7.2 -> goal 8 (15.2 m). Scale length x5, keep width.
X_WALL = 50.0
Y_WALL = 10.0
SPAWN = (-36.0, 0.0)
GOAL = (40.0, 0.0)
POLE_WEST = (-38.0, 0.0)
POLE_EAST = (40.0, 0.0)
WALL_T = 0.25
WALL_H = 4.0
N_OBS = 80
SEED = 0
CLEAR = 5.0


def _wall(name: str, x: float, y: float, sx: float, sy: float) -> str:
    return f"""    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {0.5 * WALL_H:.3f} 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>{sx:.3f} {sy:.3f} {WALL_H:.3f}</size></box></geometry></collision>
        <visual name="v">
          <geometry><box><size>{sx:.3f} {sy:.3f} {WALL_H:.3f}</size></box></geometry>
          <material><ambient>0.55 0.55 0.58 1</ambient><diffuse>0.55 0.55 0.58 1</diffuse></material>
        </visual>
      </link>
    </model>
"""


def _box(name: str, x: float, y: float, sx: float, sy: float) -> str:
    return f"""    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} 1 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><box><size>{sx:.3f} {sy:.3f} 2</size></box></geometry></collision>
        <visual name="v">
          <geometry><box><size>{sx:.3f} {sy:.3f} 2</size></box></geometry>
          <material><ambient>0.44 0.40 0.36 1</ambient><diffuse>0.44 0.40 0.36 1</diffuse></material>
        </visual>
      </link>
    </model>
"""


def _cyl(name: str, x: float, y: float, r: float) -> str:
    return f"""    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} 1 0 0 0</pose>
      <link name="link">
        <collision name="c"><geometry><cylinder><radius>{r:.3f}</radius><length>2</length></cylinder></geometry></collision>
        <visual name="v">
          <geometry><cylinder><radius>{r:.3f}</radius><length>2</length></cylinder></geometry>
          <material><ambient>0.40 0.40 0.45 1</ambient><diffuse>0.40 0.40 0.45 1</diffuse></material>
        </visual>
      </link>
    </model>
"""


def _pole(name: str, x: float, y: float, rgb: tuple[float, float, float]) -> str:
    r, g, b = rgb
    e = (0.5 * r, 0.5 * g, 0.5 * b)
    return f"""    <model name="{name}">
      <static>true</static>
      <pose>{x:.3f} {y:.3f} 0 0 0 0</pose>
      <link name="link">
        <visual name="mast">
          <pose>0 0 1.5 0 0 0</pose>
          <geometry><cylinder><radius>0.12</radius><length>3.0</length></cylinder></geometry>
          <material>
            <ambient>{r} {g} {b} 1</ambient>
            <diffuse>{r} {g} {b} 1</diffuse>
            <emissive>{e[0]} {e[1]} {e[2]} 1</emissive>
          </material>
        </visual>
        <visual name="ball">
          <pose>0 0 3.15 0 0 0</pose>
          <geometry><sphere><radius>0.28</radius></sphere></geometry>
          <material>
            <ambient>{r} {g} {b} 1</ambient>
            <diffuse>{r} {g} {b} 1</diffuse>
            <emissive>{e[0] * 1.25} {e[1] * 1.25} {e[2] * 1.25} 1</emissive>
          </material>
        </visual>
      </link>
    </model>
"""


def layout(rng: np.random.Generator) -> list[str]:
    chunks: list[str] = []
    lx = 2.0 * X_WALL
    ly = 2.0 * Y_WALL
    chunks.append(_wall("wall_north", 0.0, Y_WALL, lx + WALL_T, WALL_T))
    chunks.append(_wall("wall_south", 0.0, -Y_WALL, lx + WALL_T, WALL_T))
    chunks.append(_wall("wall_east", X_WALL, 0.0, WALL_T, ly + WALL_T))
    chunks.append(_wall("wall_west", -X_WALL, 0.0, WALL_T, ly + WALL_T))
    placed: list[np.ndarray] = []
    inner_x = X_WALL - 1.5
    inner_y = Y_WALL - 1.5
    n = 0
    attempts = 0
    while n < N_OBS and attempts < 8000:
        attempts += 1
        xy = np.array(
            [
                float(rng.uniform(-inner_x, inner_x)),
                float(rng.uniform(-inner_y, inner_y)),
            ]
        )
        if np.hypot(xy[0] - SPAWN[0], xy[1] - SPAWN[1]) < CLEAR:
            continue
        if np.hypot(xy[0] - GOAL[0], xy[1] - GOAL[1]) < CLEAR:
            continue
        if np.hypot(xy[0] - POLE_WEST[0], xy[1] - POLE_WEST[1]) < CLEAR:
            continue
        if any(np.hypot(xy[0] - p[0], xy[1] - p[1]) < 1.15 for p in placed):
            continue
        if rng.random() < 0.5:
            sx = float(rng.uniform(0.7, 1.3))
            sy = float(rng.uniform(0.7, 1.3))
            if np.hypot(xy[0] - SPAWN[0], xy[1] - SPAWN[1]) < CLEAR + 0.5 * max(sx, sy):
                continue
            placed.append(xy)
            chunks.append(_box(f"box_{n}", xy[0], xy[1], sx, sy))
        else:
            rad = float(rng.uniform(0.25, 0.42))
            if np.hypot(xy[0] - SPAWN[0], xy[1] - SPAWN[1]) < CLEAR + rad:
                continue
            placed.append(xy)
            chunks.append(_cyl(f"cyl_{n}", xy[0], xy[1], rad))
        n += 1
    chunks.append(_pole("goal_west", POLE_WEST[0], POLE_WEST[1], (0.15, 0.85, 0.25)))
    chunks.append(_pole("goal_east", POLE_EAST[0], POLE_EAST[1], (0.15, 0.35, 0.95)))
    return chunks


def main() -> None:
    rng = np.random.default_rng(SEED)
    models = "".join(layout(rng))
    sdf = f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <world name="explore">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system" name="gz::sim::systems::Contact"/>
    <plugin filename="gz-sim-imu-system" name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <gui fullscreen="0">
      <camera name="user_camera">
        <pose>-42 -14 12 0 0.55 0.75</pose>
        <view_controller>orbit</view_controller>
        <projection_type>perspective</projection_type>
      </camera>
    </gui>
    <gravity>0 0 -9.8</gravity>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 20 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.5 0.1 -0.9</direction>
    </light>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>120 40</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>120 40</size></plane></geometry>
          <material>
            <ambient>0.75 0.75 0.72 1</ambient>
            <diffuse>0.75 0.75 0.72 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
{models}  </world>
</sdf>
"""
    OUT.write_text(sdf)
    print(f"wrote {OUT} spawn={SPAWN} goal={GOAL} n_obs={N_OBS}")


if __name__ == "__main__":
    main()
