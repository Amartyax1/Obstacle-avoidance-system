# Obstacle-avoidance-system
# Obstacle avoidance for a simulated quadcopter
End-to-end RL drone navigation: PPO trained in PyBullet, exported to ONNX, and
run at 50 Hz inside a ROS 2 node that drives a Gazebo quadcopter.
A PPO policy trained in PyBullet flies a Gazebo quadcopter. The actor is exported to ONNX and runs at 50 Hz in a ROS 2 node. A live occupancy A* co-pilot feeds it a short lookahead goal; the policy still owns `/cmd_vel`.
```
PyBullet + PPO  ->  drone_brain.onnx  ->  ROS 2 node  ->  Gazebo
   (train/)          (Phase 3)          (src/drone_robot)
PyBullet + PPO  ->  drone_brain.onnx  ->  ROS 2 (50 Hz)  ->  Gazebo Harmonic
   train/         models/drone_brain.onnx   src/drone_robot
```
| Phase | Scope | Result |
|---|---|---|
| 1 | PPO navigation, 1.5M steps, 8 envs | 19/20 success, 1/20 crash |
| 2 | Fine-tune under LiDAR noise and gusts | 20/20 success under domain rand |
| 3 | ONNX export + SB3 numerical check | max error 4.8e-7 |
| 4 | ROS 2 preprocess + 50 Hz inference node | package + unit tests |
| 5 | Full Gazebo closed-loop verification | first flight reached (4, 0) in 8.4 s by climbing over 2 m cylinders (`z=2.12`). Now: training altitude hold, 8×8 m trunks, 16-ray lidar. Humble `ros_gz` cannot talk to gz-sim8 so spawn + bridge use native `gz.transport` |
| Factory baseline | Phase 2 forest ONNX, west `(-5,0)` → east `(26,0)` → west, 90 s | **FAIL**: east MISS (closest 27.9 m), x only `-5.00 → -1.88`, z `0.87–1.00`, no crash. Factory RTF is low (~1.5 Hz odom); policy crawled but did not transit. |
| Warehouse PPO | In-domain URDF, 1.5M steps | PyBullet **12/20** success (A* co-pilot covers the rest) |
| Piece | Role |
|---|---|
| `train/` | Standalone PyBullet training venv. See [train/README.md](train/README.md). |
| `src/drone_robot/` | ROS 2 package: worlds, model, Gazebo bridge, inference, A* planner. |
| `src/drone_robot/models/drone_brain.onnx` | Exported actor, clipped to the action bounds. |
## Layout
## Requirements
- `train/` — standalone PyBullet training venv (see [train/README.md](train/README.md))
- `src/drone_robot/` — ROS 2 package: model, world, bridge, inference node
- `src/drone_robot/models/drone_brain.onnx` — corridor PPO actor (17/20 PyBullet), clipped to the action bounds
- ROS 2 Humble
- Gazebo Harmonic (`gz sim` / gz-sim8) and `python3-gz-transport13`
- Python 3.10
## Observation contract
Humble `ros_gz_bridge` and `ros_gz_sim create` cannot talk to gz-sim8 (`Unknown message type [8]`). Launch starts `gz sim` itself, spawns the drone with `scripts/spawn_drone.py`, and bridges `/scan`, `/odom`, `/clock`, and `/cmd_vel` through `gz_bridge_node.py`.
18 floats: 16 LiDAR rays over the forward 180 deg (0–5 m, mapped to `[-1, 1]`)
plus the body-frame goal vector divided by **5 m**. Actions are `vx` in
`[-0.5, 1.5]` m/s and `wz` in `[-1, 1]` rad/s.
ONNX Runtime has no rosdep key on Humble:
Training (`train/envs/lidar_nav_env.py`) and deployment
(`src/drone_robot/drone_robot/preprocess.py`) must agree on this or the policy
sees out-of-distribution input.
## Run it
```bash
# one-time: ONNX Runtime has no rosdep key on Humble
pip3 install --user onnxruntime
```
## Run
```bash
source /opt/ros/humble/setup.bash
cd ~/drone_ws
colcon build --symlink-install --packages-select drone_robot
source install/setup.bash
# first time only: python3 train/scripts/generate_warehouse.py --seed 0 --name warehouse --eval-suite
ros2 launch drone_robot ros2launch.py
# A* GPS co-pilot + RViz path:
# ros2 launch drone_robot ros2launch.py use_planner:=true use_rviz:=true loop_waypoints:=false
# five unseen worlds (pause before each for recording):
# src/drone_robot/scripts/run_gazebo_evals.sh
```
Gazebo Harmonic (`gz-sim8`) is on this machine; Humble `ros_gz_bridge` / `ros_gz_sim create` fail with `Unknown message type [8]`. Launch therefore:
That starts the default `explore` world: a long west–east corridor, spawn at `(-36, 0, 1)`, goal at `(40, 0)`. The planner is on and RViz is off.
1. starts `gz sim` on `worlds/warehouse.sdf` (procedural boxes; `world:=forest` / `world:=factory` still work)
2. spawns the xacro via `scripts/spawn_drone.py` at the west end `(-5, 0, 1)`
3. bridges `/scan`, `/odom`, `/clock`, `/cmd_vel` with `gz_bridge_node.py` (`python3-gz-transport13`)
4. runs the ONNX node at 50 Hz, cycling waypoints `(-5, 0)` ↔ `(26, 0)`
5. optional `use_planner:=true` starts A* which **only** publishes `/lookahead_goal` (body-frame transform in inference); it never writes `/cmd_vel`
```bash
# warehouse boxes, west spawn, optional RViz path
ros2 launch drone_robot ros2launch.py \
  world:=warehouse spawn_x:=-5.0 use_rviz:=true
One-time factory meshes (kept out of git, 29 MB):
# policy only, no A* lookahead
ros2 launch drone_robot ros2launch.py use_planner:=false
```
| Argument | Default | Meaning |
|---|---|---|
| `world` | `explore` | Stem under `worlds/`: `explore`, `warehouse`, `eval_1`–`eval_5`, `forest`, `factory` |
| `spawn_x`, `spawn_y`, `spawn_z` | `-36`, `0`, `1` | Spawn pose. Warehouse and eval worlds use `spawn_x:=-5.0`. |
| `use_planner` | `true` | Live occupancy A*. Publishes `/lookahead_goal` only. |
| `use_rviz` | `false` | Global path and lookahead marker (`config/planner.rviz`). |
| `loop_waypoints` | `false` | After the last goal, cycle back to the first. |
| `headless` | `false` | `gz sim -s` with no GUI. |
`explore` overrides the waypoint list to a single goal at `(40, 0)` and widens the planner grid to about `x ∈ [-52, 52]`, `y ∈ [-12, 12]`. Other worlds keep the params in `config/params.yaml`: east goal `(26, 0)`, then west `(-5, 0)` when looping.
Regenerate the explore world:
```bash
python3 train/scripts/generate_explore.py
```
Warehouse and the five held-out eval worlds (first time, or after a layout change):
```bash
python3 train/scripts/generate_warehouse.py --seed 0 --name warehouse --eval-suite
```
Factory meshes stay out of git (~29 MB):
```bash
src/drone_robot/scripts/install_factory_world.sh
# same as: export GAZEBO_MODEL_PATH=$HOME/.gazebo/models/factory/models
```
Phase 5 check (2026-09-22): first run reached `(4, 0)` in 8.4 s at `z=2.12` by flying over 2 m cylinders (`/cmd_vel` had no `vz`). After hover hold + 8 m trunks + 16-ray lidar: **z stays in 0.98–1.27 m**, the drone weaves instead of climbing over, closest approach to `(4, 0)` was **0.73 m** (then it circled). A packed 14-tree FOV made this PPO actor output `vx=0`.
## What runs
Factory baseline (2026-09-22, `bench_factory.py`, 90 s, `loop_waypoints:=false`): spawn `(-5,0,1)`, goal east then west. **Did not reach `(26,0)`.** End pose `(-1.88, 0.17, 1.00)`, z `0.87–1.00`, no crash, 137 odom samples (sim well below real time). The Phase 2 forest policy does not complete a factory transit.
1. `gz sim` loads `worlds/<world>.sdf`.
2. `robot_state_publisher` publishes the xacro in `model/robot.xacro`.
3. `gz_bridge_node.py` bridges Gazebo transport to ROS 2.
4. After 3 s, `spawn_drone.py` inserts the drone.
5. `inference_node.py` loads the ONNX actor and commands `/cmd_vel` at 50 Hz.
6. With `use_planner:=true`, `planner_node.py` carves an optimistic occupancy grid from `/scan` (unknown cells are free) and publishes a 2 m lookahead on `/lookahead_goal`. It never writes `/cmd_vel`.
Factory after corridor training (2026-09-22, same protocol, new ONNX): PyBullet corridor eval **17/20** (1.5M scratch + 750k). Gazebo **still FAIL** — east closest 30.3 m, x `-4.90 → -4.34`, z `0.94–1.01`, freeze 6.0 s, 166 odom samples in 90 s. The factory GPU lidar keeps sim at ~2 Hz, so 90 s wall is only a few seconds of flight; when the policy did run it yawed (`wz=-1`) in a filled FOV. Next knob is FOV density, not more PPO steps.
The policy outputs body-frame `vx` and `wz`. Altitude is a separate PD hold at `z = 1 m` (`vz` clipped to `±1` m/s), matching training. If `/scan` or `/odom` is older than 0.5 s the node stops commanding instead of flying blind. A stale lookahead (older than 1 s) falls back to the waypoint goal.
Closed-loop smoke test without Gazebo:
## Observation and action contract
Training (`train/envs/lidar_nav_env.py`) and deployment (`src/drone_robot/drone_robot/preprocess.py`) share one contract. If they diverge, the policy sees out-of-distribution input.
| Item | Spec |
|---|---|
| Observation | 18 floats in `[-1, 1]` |
| LiDAR | 16 rays, forward 180°, 0–5 m, mapped to `[-1, 1]` |
| Goal | Body-frame `(dx, dy)` divided by 5 m, clipped to `[-1, 1]` |
| Action | `vx` in `[-0.5, 1.5]` m/s, `wz` in `[-1, 1]` rad/s |
| Altitude | PD hold at 1 m; not part of the actor |
With the planner on, the goal vector is the lookahead pose, not the global waypoint.
## Train and export
Do this in a shell that has **not** sourced ROS. The training venv pins its own NumPy and PyTorch.
```bash
cd ~/drone_ws/train
bash scripts/setup_venv.sh
source .venv/bin/activate
python sanity_check.py
python train_ppo.py --config configs/ppo_nav.yaml
python export_onnx.py --ckpt runs/phase2_dr/latest.zip \
    --out ../src/drone_robot/models/drone_brain.onnx
```
Configs: `ppo_nav.yaml` (forest), `ppo_nav_dr.yaml` (noise and gusts), `ppo_nav_factory.yaml`, `ppo_nav_warehouse.yaml`, `ppo_nav_explore.yaml`. Details, eval commands, and the NumPy `<2` pin are in [train/README.md](train/README.md). `runs/` and `.venv/` are gitignored; the exported `.onnx` is what deployment uses.
## Checks
Closed loop without Gazebo (synthetic scan and odometry, real ONNX node):
```bash
source /opt/ros/humble/setup.bash
source ~/drone_ws/install/setup.bash
python3 src/drone_robot/scripts/check_inference_loop.py
```
Unit tests:
Package tests:
```bash
colcon test --packages-select drone_robot && colcon test-result --all
```
`xmllint` needs network access to fetch the ROS package schema; it fails
offline and that failure is not a code problem.
`xmllint` fetches the ROS package schema and fails offline. That failure is not a code problem.
Interactive Gazebo eval on `eval_1` … `eval_5` (pauses for recording, writes `generalization_results.csv`):
```bash
src/drone_robot/scripts/run_gazebo_evals.sh
```
