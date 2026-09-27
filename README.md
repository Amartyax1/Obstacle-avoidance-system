# Obstacle-avoidance-system

End-to-end RL drone navigation: PPO trained in PyBullet, exported to ONNX, and
run at 50 Hz inside a ROS 2 node that drives a Gazebo quadcopter.

```
PyBullet + PPO  ->  drone_brain.onnx  ->  ROS 2 node  ->  Gazebo
   (train/)          (Phase 3)          (src/drone_robot)
```

| Phase | Scope | Result |
|---|---|---|
| 1 | PPO navigation, 1.5M steps, 8 envs | 19/20 success, 1/20 crash |
| 2 | Fine-tune under LiDAR noise and gusts | 20/20 success under domain rand |
| 3 | ONNX export + SB3 numerical check | max error 4.8e-7 |
| 4 | ROS 2 preprocess + 50 Hz inference node | package + unit tests |
| 5 | Full Gazebo closed-loop verification | first flight reached (4, 0) in 8.4 s by climbing over 2 m cylinders (`z=2.12`). Now: training altitude hold, 8×8 m trunks, 16-ray lidar. Humble `ros_gz` cannot talk to gz-sim8 so spawn + bridge use native `gz.transport` |
| Factory baseline | Phase 2 forest ONNX, west `(-5,0)` → east `(26,0)` → west, 90 s | **FAIL**: east MISS (closest 27.9 m), x only `-5.00 → -1.88`, z `0.87–1.00`, no crash. Factory RTF is low (~1.5 Hz odom); policy crawled but did not transit. |
| Factory after train | Corridor PPO 17/20 in PyBullet, same 90 s Gazebo protocol | **FAIL**: east MISS (closest 30.3 m), x `-4.90 → -4.34`, z `0.94–1.01`, freeze 6.0 s, no crash. 166 odom samples in 90 s (~1.8 Hz). Policy commanded `vx≈0.6` but `wz=-1` (yawing in a dense FOV). Next knob is FOV density, not more PPO steps. |

## Layout

- `train/` — standalone PyBullet training venv (see [train/README.md](train/README.md))
- `src/drone_robot/` — ROS 2 package: model, world, bridge, inference node
- `src/drone_robot/models/drone_brain.onnx` — corridor PPO actor (17/20 PyBullet), clipped to the action bounds

## Observation contract

18 floats: 16 LiDAR rays over the forward 180 deg (0–5 m, mapped to `[-1, 1]`)
plus the body-frame goal vector divided by **5 m**. Actions are `vx` in
`[-0.5, 1.5]` m/s and `wz` in `[-1, 1]` rad/s.

Training (`train/envs/lidar_nav_env.py`) and deployment
(`src/drone_robot/drone_robot/preprocess.py`) must agree on this or the policy
sees out-of-distribution input.

## Run it

```bash
# one-time: ONNX Runtime has no rosdep key on Humble
pip3 install --user onnxruntime

source /opt/ros/humble/setup.bash
cd ~/drone_ws
colcon build --symlink-install --packages-select drone_robot
source install/setup.bash
# first time only: src/drone_robot/scripts/install_factory_world.sh
ros2 launch drone_robot ros2launch.py
```

Gazebo Harmonic (`gz-sim8`) is on this machine; Humble `ros_gz_bridge` / `ros_gz_sim create` fail with `Unknown message type [8]`. Launch therefore:

1. starts `gz sim` on `worlds/factory.sdf` (mlherd factory; `world:=forest` still works)
2. spawns the xacro via `scripts/spawn_drone.py` at the west end `(-5, 0, 1)`
3. bridges `/scan`, `/odom`, `/clock`, `/cmd_vel` with `gz_bridge_node.py` (`python3-gz-transport13`)
4. runs the ONNX node at 50 Hz, cycling waypoints `(-5, 0)` ↔ `(26, 0)` through the factory

One-time factory meshes (kept out of git, 29 MB):

```bash
src/drone_robot/scripts/install_factory_world.sh
# same as: export GAZEBO_MODEL_PATH=$HOME/.gazebo/models/factory/models
```

Phase 5 check (2026-09-22): first run reached `(4, 0)` in 8.4 s at `z=2.12` by flying over 2 m cylinders (`/cmd_vel` had no `vz`). After hover hold + 8 m trunks + 16-ray lidar: **z stays in 0.98–1.27 m**, the drone weaves instead of climbing over, closest approach to `(4, 0)` was **0.73 m** (then it circled). A packed 14-tree FOV made this PPO actor output `vx=0`.

Factory baseline (2026-09-22, `bench_factory.py`, 90 s, `loop_waypoints:=false`): spawn `(-5,0,1)`, goal east then west. **Did not reach `(26,0)`.** End pose `(-1.88, 0.17, 1.00)`, z `0.87–1.00`, no crash, 137 odom samples (sim well below real time). The Phase 2 forest policy does not complete a factory transit.

Factory after corridor training (2026-09-22, same protocol, new ONNX): PyBullet corridor eval **17/20** (1.5M scratch + 750k). Gazebo **still FAIL** — east closest 30.3 m, x `-4.90 → -4.34`, z `0.94–1.01`, freeze 6.0 s, 166 odom samples in 90 s. The factory GPU lidar keeps sim at ~2 Hz, so 90 s wall is only a few seconds of flight; when the policy did run it yawed (`wz=-1`) in a filled FOV. Next knob is FOV density, not more PPO steps.

Closed-loop smoke test without Gazebo:

```bash
source /opt/ros/humble/setup.bash
source ~/drone_ws/install/setup.bash
python3 src/drone_robot/scripts/check_inference_loop.py
```

Unit tests:

```bash
colcon test --packages-select drone_robot && colcon test-result --all
```

`xmllint` needs network access to fetch the ROS package schema; it fails
offline and that failure is not a code problem.
