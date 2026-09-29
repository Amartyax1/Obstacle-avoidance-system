"""Planar LiDAR navigation env wrapping gym-pybullet-drones VelocityAviary.

Observation (18,): 16 forward rays in [-1, 1] plus body-frame goal (dx, dy).
Action (2,): forward speed vx in [-0.5, 1.5] m/s and yaw rate wz in [-1, 1] rad/s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from .online_occupancy import (
    GRID_RES,
    INFLATE_M,
    astar,
    carve_rays,
    lookahead_on_path,
    make_grid,
)

N_RAYS = 16
OBS_DIM = 18
MAX_RANGE = 5.0
HOVER_Z = 1.0
VX_MIN, VX_MAX = -0.5, 1.5
WZ_MIN, WZ_MAX = -1.0, 1.0
GOAL_RADIUS = 0.5
OBSTACLE_RADIUS = 0.25

# The attitude controller tracks a yaw setpoint, so a rate command has to be
# projected this far ahead to produce usable turn authority.
YAW_LOOKAHEAD_SEC = 0.3
# Distance at which the shaping term starts pushing the drone away from clutter.
# Kept tight on purpose: with 8 cylinders a wider zone covers most of the arena
# and turns into a constant drag instead of a signal. The penalty is quadratic,
# so brushing past is nearly free and closing in is expensive.
CLEARANCE_M = 1.0
CLEARANCE_WEIGHT = 2.0

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_WAREHOUSE_URDF = _REPO / "src" / "drone_robot" / "models" / "warehouse.urdf"
DEFAULT_WAREHOUSE_NPZ = _REPO / "src" / "drone_robot" / "config" / "warehouse_grid.npz"


class LidarNavEnv(VelocityAviary):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        gui: bool = False,
        domain_rand: bool = False,
        n_obstacles: int = 8,
        world_size: float = 8.0,
        episode_len_sec: float = 20.0,
        lidar_noise_std: float = 0.05,
        layout: str = "forest",
        warehouse_urdf: str | None = None,
        seed: int | None = None,
    ):
        self.domain_rand = domain_rand
        self.n_obstacles = n_obstacles
        self.world_size = world_size
        self.episode_len_sec = episode_len_sec
        self.layout = layout
        self.warehouse_urdf = Path(warehouse_urdf) if warehouse_urdf else DEFAULT_WAREHOUSE_URDF
        self.corridor_half_w = 3.0
        self.lidar_noise_std = lidar_noise_std if domain_rand else 0.0
        self.goal_xy = np.zeros(2, dtype=np.float32)
        self._lookahead_xy = np.zeros(2, dtype=np.float32)
        self.obstacle_ids: list[int] = []
        self.obstacle_xy: np.ndarray = np.zeros((0, 2))
        self._box_obstacles: list[tuple[float, float, float, float]] = []
        self._occ: np.ndarray | None = None
        self._xs: np.ndarray | None = None
        self._ys: np.ndarray | None = None
        self._prev_goal_dist = 0.0
        self._wind_step = 0
        self._target_yaw = 0.0
        self._min_range = MAX_RANGE
        self._rng = np.random.default_rng(seed)

        init_xyzs = np.array([[0.0, 0.0, HOVER_Z]])
        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=init_xyzs,
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=30,
            gui=gui,
            record=False,
            obstacles=True,
            user_debug_gui=False,
        )
        self.SPEED_LIMIT = float(VX_MAX)
        self.action_space = spaces.Box(
            low=np.array([VX_MIN, WZ_MIN], dtype=np.float32),
            high=np.array([VX_MAX, WZ_MAX], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )

    def _actionSpace(self):
        return spaces.Box(
            low=np.array([VX_MIN, WZ_MIN], dtype=np.float32),
            high=np.array([VX_MAX, WZ_MAX], dtype=np.float32),
            dtype=np.float32,
        )

    def _observationSpace(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._layout_episode()
        obs, _ = super().reset(seed=seed, options=options)
        state = self._getDroneStateVector(0)
        self._target_yaw = float(state[9])
        self._prev_goal_dist = self._goal_distance(state)
        self._wind_step = 0
        return obs, self._computeInfo()

    def _layout_episode(self):
        if self.layout == "warehouse":
            self._layout_warehouse()
            return
        if self.layout == "corridor":
            self._layout_corridor()
            return
        if self.layout == "explore":
            self._layout_explore()
            return
        half = self.world_size * 0.45
        spawn = self._rng.uniform(-half * 0.4, half * 0.4, size=2)
        yaw = float(self._rng.uniform(-np.pi, np.pi))
        self.INIT_XYZS = np.array([[spawn[0], spawn[1], HOVER_Z]], dtype=np.float64)
        self.INIT_RPYS = np.array([[0.0, 0.0, yaw]], dtype=np.float64)

        for _ in range(32):
            goal = self._rng.uniform(-half, half, size=2)
            if np.linalg.norm(goal - spawn) > 3.0:
                self.goal_xy = goal.astype(np.float32)
                break
        else:
            self.goal_xy = np.array([half, half], dtype=np.float32)

        cylinders: list[np.ndarray] = []
        attempts = 0
        while len(cylinders) < self.n_obstacles and attempts < 512:
            attempts += 1
            xy = self._rng.uniform(-half, half, size=2)
            if np.linalg.norm(xy - spawn) < 1.2:
                continue
            if np.linalg.norm(xy - self.goal_xy) < 1.2:
                continue
            if any(np.linalg.norm(xy - c) < 1.0 for c in cylinders):
                continue
            cylinders.append(xy)
        self.obstacle_xy = np.asarray(cylinders, dtype=np.float32)
        self._box_obstacles = []

    def _layout_explore(self):
        half = 0.5 * float(self.world_size)
        inset = 2.0
        inner = max(half - inset, 1.0)
        go_east = bool(self._rng.random() < 0.5)
        y_spawn = float(self._rng.uniform(-inner, inner))
        y_goal = float(self._rng.uniform(-inner, inner))
        west = np.array([-inner, y_spawn], dtype=np.float64)
        east = np.array([inner, y_goal], dtype=np.float64)
        spawn = west if go_east else east
        goal = east if go_east else west
        yaw = 0.0 if go_east else float(np.pi)
        self.INIT_XYZS = np.array([[spawn[0], spawn[1], HOVER_Z]], dtype=np.float64)
        self.INIT_RPYS = np.array([[0.0, 0.0, yaw]], dtype=np.float64)
        self.goal_xy = goal.astype(np.float32)
        self._lookahead_xy = self.goal_xy.copy()

        n = int(self._rng.integers(5, 16))
        cylinders: list[np.ndarray] = []
        boxes: list[tuple[float, float, float, float]] = []
        attempts = 0
        while (len(cylinders) + len(boxes)) < n and attempts < 768:
            attempts += 1
            xy = self._rng.uniform(-inner, inner, size=2)
            if np.linalg.norm(xy - spawn) < 1.5:
                continue
            if np.linalg.norm(xy - goal) < 1.5:
                continue
            placed = list(cylinders) + [np.array([b[0], b[1]]) for b in boxes]
            if any(np.linalg.norm(xy - c) < 1.1 for c in placed):
                continue
            if self._rng.random() < 0.5:
                cylinders.append(xy)
            else:
                hx = float(self._rng.uniform(0.25, 0.7))
                hy = float(self._rng.uniform(0.25, 0.7))
                boxes.append((float(xy[0]), float(xy[1]), hx, hy))
        self.obstacle_xy = np.asarray(cylinders, dtype=np.float32)
        self._box_obstacles = boxes
        pad = 0.5
        self._occ, self._xs, self._ys = make_grid(
            -half - pad, half + pad, -half - pad, half + pad, GRID_RES
        )

    def _layout_warehouse(self):
        npz = DEFAULT_WAREHOUSE_NPZ
        name = self.warehouse_urdf.stem
        candidate = npz.with_name(f"{name}_grid.npz")
        if candidate.is_file():
            npz = candidate
        data = np.load(npz, allow_pickle=True)
        spawn = np.asarray(data["spawn"], dtype=np.float64)
        goal = np.asarray(data["goal"], dtype=np.float64)
        go_east = bool(self._rng.random() < 0.5)
        west = spawn + np.array([0.0, float(self._rng.uniform(-0.35, 0.35))])
        east = goal + np.array([0.0, float(self._rng.uniform(-0.35, 0.35))])
        start = west if go_east else east
        end = east if go_east else west
        yaw = 0.0 if go_east else float(np.pi)
        self.INIT_XYZS = np.array([[start[0], start[1], HOVER_Z]], dtype=np.float64)
        self.INIT_RPYS = np.array([[0.0, 0.0, yaw]], dtype=np.float64)
        self.goal_xy = end.astype(np.float32)
        self.obstacle_xy = np.zeros((0, 2), dtype=np.float32)

    def _layout_corridor(self):
        length = float(self.world_size)
        west = np.array([1.5, float(self._rng.uniform(-0.4, 0.4))])
        east = np.array([length - 1.5, float(self._rng.uniform(-0.4, 0.4))])
        go_east = bool(self._rng.random() < 0.5)
        spawn = west if go_east else east
        goal = east if go_east else west
        yaw = 0.0 if go_east else float(np.pi)
        self.INIT_XYZS = np.array([[spawn[0], spawn[1], HOVER_Z]], dtype=np.float64)
        self.INIT_RPYS = np.array([[0.0, 0.0, yaw]], dtype=np.float64)
        self.goal_xy = goal.astype(np.float32)

        # Stagger trunks along the aisle so a center lane stays open. Pure 2-D
        # rejection sampling with 12 disks in a 6 m hall can wall off the path.
        n = max(int(self.n_obstacles), 0)
        xs = np.linspace(4.0, length - 4.0, num=max(n, 1))
        cylinders: list[np.ndarray] = []
        for i, x in enumerate(xs[:n]):
            side = 1.0 if i % 2 == 0 else -1.0
            y = side * float(self._rng.uniform(1.15, 2.15))
            xy = np.array([float(x) + float(self._rng.uniform(-0.35, 0.35)), y])
            if np.linalg.norm(xy - spawn) < 1.2:
                continue
            if np.linalg.norm(xy - goal) < 1.2:
                continue
            if any(np.linalg.norm(xy - c) < 1.05 for c in cylinders):
                continue
            cylinders.append(xy)
        self.obstacle_xy = np.asarray(cylinders, dtype=np.float32)

    def _addObstacles(self):
        self.obstacle_ids = []
        if self.layout == "warehouse":
            if not self.warehouse_urdf.is_file():
                raise FileNotFoundError(
                    f"warehouse URDF missing: {self.warehouse_urdf}. "
                    "Run train/scripts/generate_warehouse.py"
                )
            uid = p.loadURDF(
                str(self.warehouse_urdf),
                [0, 0, 0],
                useFixedBase=True,
                physicsClientId=self.CLIENT,
            )
            self.obstacle_ids.append(uid)
            if self.GUI:
                p.addUserDebugLine(
                    [self.goal_xy[0], self.goal_xy[1], 0.05],
                    [self.goal_xy[0], self.goal_xy[1], 1.5],
                    [0.1, 0.8, 0.2],
                    lineWidth=3,
                    physicsClientId=self.CLIENT,
                )
            return
        if len(self.obstacle_xy) > 0:
            col = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=OBSTACLE_RADIUS,
                height=2.0,
                physicsClientId=self.CLIENT,
            )
            vis = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=OBSTACLE_RADIUS,
                length=2.0,
                rgbaColor=[0.4, 0.4, 0.45, 1.0],
                physicsClientId=self.CLIENT,
            )
            for xy in self.obstacle_xy:
                body = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=col,
                    baseVisualShapeIndex=vis,
                    basePosition=[float(xy[0]), float(xy[1]), 1.0],
                    physicsClientId=self.CLIENT,
                )
                self.obstacle_ids.append(body)
        if self.layout == "explore":
            self._addExploreBoxes()
            self._addExploreWalls()
        if self.layout == "corridor":
            self._addCorridorWalls()
        if self.GUI:
            p.addUserDebugLine(
                [self.goal_xy[0], self.goal_xy[1], 0.05],
                [self.goal_xy[0], self.goal_xy[1], 1.5],
                [0.1, 0.8, 0.2],
                lineWidth=3,
                physicsClientId=self.CLIENT,
            )

    def _addCorridorWalls(self):
        length = float(self.world_size)
        thick = 0.2
        height = 4.0
        half_w = self.corridor_half_w
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[length * 0.5, thick * 0.5, height * 0.5],
            physicsClientId=self.CLIENT,
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[length * 0.5, thick * 0.5, height * 0.5],
            rgbaColor=[0.55, 0.55, 0.58, 1.0],
            physicsClientId=self.CLIENT,
        )
        for y in (half_w, -half_w):
            body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[length * 0.5, y, height * 0.5],
                physicsClientId=self.CLIENT,
            )
            self.obstacle_ids.append(body)

    def _addExploreWalls(self):
        half = 0.5 * float(self.world_size)
        thick = 0.25
        height = 4.0
        length = 2.0 * half
        col_ew = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[length * 0.5, thick * 0.5, height * 0.5],
            physicsClientId=self.CLIENT,
        )
        vis_ew = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[length * 0.5, thick * 0.5, height * 0.5],
            rgbaColor=[0.55, 0.55, 0.58, 1.0],
            physicsClientId=self.CLIENT,
        )
        col_ns = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[thick * 0.5, length * 0.5, height * 0.5],
            physicsClientId=self.CLIENT,
        )
        vis_ns = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[thick * 0.5, length * 0.5, height * 0.5],
            rgbaColor=[0.55, 0.55, 0.58, 1.0],
            physicsClientId=self.CLIENT,
        )
        for y in (half, -half):
            body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_ew,
                baseVisualShapeIndex=vis_ew,
                basePosition=[0.0, y, height * 0.5],
                physicsClientId=self.CLIENT,
            )
            self.obstacle_ids.append(body)
        for x in (half, -half):
            body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col_ns,
                baseVisualShapeIndex=vis_ns,
                basePosition=[x, 0.0, height * 0.5],
                physicsClientId=self.CLIENT,
            )
            self.obstacle_ids.append(body)

    def _addExploreBoxes(self):
        height = 2.0
        for x, y, hx, hy in self._box_obstacles:
            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[hx, hy, height * 0.5],
                physicsClientId=self.CLIENT,
            )
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[hx, hy, height * 0.5],
                rgbaColor=[0.45, 0.42, 0.38, 1.0],
                physicsClientId=self.CLIENT,
            )
            body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=[x, y, height * 0.5],
                physicsClientId=self.CLIENT,
            )
            self.obstacle_ids.append(body)

    def _preprocessAction(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        vx = float(np.clip(action[0], VX_MIN, VX_MAX))
        wz = float(np.clip(action[1], WZ_MIN, WZ_MAX))
        state = self._getDroneStateVector(0)
        yaw = float(state[9])
        z = float(state[2])
        self._target_yaw = yaw + wz * YAW_LOOKAHEAD_SEC
        vz = float(np.clip(1.8 * (HOVER_Z - z), -1.0, 1.0))
        world_v = np.array([vx * np.cos(yaw), vx * np.sin(yaw), vz], dtype=np.float64)
        speed = float(np.linalg.norm(world_v))
        if speed < 1e-6:
            v_unit = np.zeros(3)
            frac = 0.0
        else:
            v_unit = world_v / speed
            frac = float(np.clip(speed / self.SPEED_LIMIT, 0.0, 1.0))

        rpm = np.zeros((self.NUM_DRONES, 4))
        temp, _, _ = self.ctrl[0].computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=state[0:3],
            cur_quat=state[3:7],
            cur_vel=state[10:13],
            cur_ang_vel=state[13:16],
            target_pos=state[0:3],
            target_rpy=np.array([0.0, 0.0, self._target_yaw]),
            target_vel=self.SPEED_LIMIT * frac * v_unit,
        )
        rpm[0, :] = temp
        return rpm

    def _saveLastAction(self, action):
        # BaseAviary assumes 4 RPMs; this env commands 2 planar velocities.
        padded = np.zeros((self.NUM_DRONES, 4), dtype=np.float32)
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        padded[0, : min(4, a.size)] = a[:4]
        self.last_action = padded

    def _cast_lidar(self, state):
        pos = state[0:3]
        yaw = float(state[9])
        origin = pos + np.array([0.0, 0.0, 0.02])
        headings = self._lidar_headings(yaw)
        directions = np.stack(
            [np.cos(headings), np.sin(headings), np.zeros(N_RAYS)], axis=1
        )
        ray_from = np.tile(origin, (N_RAYS, 1))
        ray_to = ray_from + MAX_RANGE * directions
        hits = p.rayTestBatch(
            ray_from.tolist(), ray_to.tolist(), physicsClientId=self.CLIENT
        )
        ranges = np.full(N_RAYS, MAX_RANGE, dtype=np.float32)
        drone_uid = int(self.DRONE_IDS[0])
        for i, hit in enumerate(hits):
            if hit[0] >= 0 and hit[0] != drone_uid:
                ranges[i] = float(hit[2] * MAX_RANGE)
        if self.lidar_noise_std > 0.0:
            ranges = ranges + self._rng.normal(
                0.0, self.lidar_noise_std, size=N_RAYS
            ).astype(np.float32)
        return np.clip(ranges, 0.0, MAX_RANGE)

    def _lidar_headings(self, yaw: float) -> np.ndarray:
        angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, N_RAYS)
        return yaw + angles

    def _body_xy(self, state, target_xy):
        yaw = float(state[9])
        world = np.asarray(target_xy, dtype=np.float32) - state[0:2]
        c, s = np.cos(yaw), np.sin(yaw)
        dx = c * world[0] + s * world[1]
        dy = -s * world[0] + c * world[1]
        return np.array([dx, dy], dtype=np.float32)

    def _body_goal(self, state):
        return self._body_xy(state, self.goal_xy)

    def _computeObs(self):
        state = self._getDroneStateVector(0)
        ranges = self._cast_lidar(state)
        self._min_range = float(np.min(ranges))
        rays = (2.0 * (ranges / MAX_RANGE) - 1.0).astype(np.float32)
        target = self.goal_xy
        if self.layout == "explore" and self._occ is not None:
            pose_xy = (float(state[0]), float(state[1]))
            goal_xy = (float(self.goal_xy[0]), float(self.goal_xy[1]))
            headings = self._lidar_headings(float(state[9]))
            carve_rays(
                self._occ,
                self._xs,
                self._ys,
                pose_xy,
                ranges,
                headings,
                MAX_RANGE,
                inflate_m=INFLATE_M,
            )
            path = astar(self._occ, self._xs, self._ys, pose_xy, goal_xy)
            look = lookahead_on_path(path, pose_xy, distance=2.0)
            if look is None:
                look = goal_xy
            self._lookahead_xy = np.array(look, dtype=np.float32)
            target = self._lookahead_xy
        goal = np.clip(self._body_xy(state, target) / MAX_RANGE, -1.0, 1.0)
        return np.concatenate([rays, goal]).astype(np.float32)

    def _goal_distance(self, state=None):
        if state is None:
            state = self._getDroneStateVector(0)
        return float(np.linalg.norm(self.goal_xy - state[0:2]))

    def _collided(self, state=None):
        if state is None:
            state = self._getDroneStateVector(0)
        if state[2] < 0.15 or state[2] > 2.5:
            return True
        if self.layout == "warehouse":
            if state[0] < -8.5 or state[0] > 30.5 or abs(state[1]) > 12.5:
                return True
        elif self.layout == "corridor":
            if state[0] < -0.5 or state[0] > self.world_size + 0.5:
                return True
            if abs(state[1]) > self.corridor_half_w - 0.05:
                return True
        elif self.layout == "explore":
            limit = 0.5 * float(self.world_size) - 0.18
            if abs(state[0]) > limit or abs(state[1]) > limit:
                return True
        elif abs(state[0]) > self.world_size or abs(state[1]) > self.world_size:
            return True
        contacts = p.getContactPoints(
            bodyA=int(self.DRONE_IDS[0]), physicsClientId=self.CLIENT
        )
        return any(c[2] in self.obstacle_ids for c in contacts)

    def _maybe_wind(self):
        if not self.domain_rand:
            return
        self._wind_step += 1
        if self._wind_step % 100 != 0:
            return
        # Scale with weight so the gust perturbs the drone instead of throwing it.
        scale = 0.05 * self.M * self.G
        force = self._rng.normal(0.0, scale, size=3)
        force[2] = 0.0
        p.applyExternalForce(
            int(self.DRONE_IDS[0]),
            -1,
            forceObj=force.tolist(),
            posObj=[0, 0, 0],
            flags=p.WORLD_FRAME,
            physicsClientId=self.CLIENT,
        )

    def _computeReward(self):
        self._maybe_wind()
        state = self._getDroneStateVector(0)
        dist = self._goal_distance(state)
        progress = self._prev_goal_dist - dist
        self._prev_goal_dist = dist

        reward = 2.0 * float(progress) - 0.02
        heading_xy = (
            self._lookahead_xy if self.layout == "explore" else self.goal_xy
        )
        body = self._body_xy(state, heading_xy)
        g_norm = float(np.linalg.norm(body)) + 1e-6
        reward += 0.1 * float(body[0] / g_norm)
        if self._min_range < CLEARANCE_M:
            # Quadratic so grazing an obstacle is cheap but closing in is not.
            encroach = (CLEARANCE_M - self._min_range) / CLEARANCE_M
            reward -= CLEARANCE_WEIGHT * encroach * encroach
        if dist < GOAL_RADIUS:
            reward += 30.0
        elif self._collided(state):
            reward -= 30.0
        return reward

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        return self._goal_distance(state) < GOAL_RADIUS or self._collided(state)

    def _computeTruncated(self):
        return self.step_counter / self.PYB_FREQ > self.episode_len_sec

    def _computeInfo(self):
        state = self._getDroneStateVector(0)
        dist = self._goal_distance(state)
        reached = dist < GOAL_RADIUS
        return {
            "goal_distance": dist,
            "min_range": self._min_range,
            "success": bool(reached),
            "collision": bool(not reached and self._collided(state)),
        }
