# Local RL training (Phases 1-3)

Standalone PyBullet stack. Do **not** source ROS Humble in this shell; the venv
ships its own NumPy/PyTorch and will conflict with `/opt/ros`.

Python 3.10 with a pinned `gym-pybullet-drones` commit (`bc83796`), because
current upstream requires Python 3.12.

## Observation / action contract (frozen)

| Item | Spec |
|---|---|
| Obs | `Box(18,)` float32 in `[-1, 1]` |
| Rays | 16 planar, 180 deg forward, 0-5 m, mapped to `[-1, 1]` |
| Goal | body-frame `(dx, dy) / 5`, clipped to `[-1, 1]` |
| Action | `vx` in `[-0.5, 1.5]` m/s, `wz` in `[-1, 1]` rad/s |
| Inner control | `VelocityAviary` PID, hover z = 1 m |

`src/drone_robot/drone_robot/preprocess.py` must mirror this exactly. The unit
test `test_goal_scaling_matches_training_range` pins the goal scale to 5 m.

## Setup (local CPU)

```bash
cd ~/drone_ws/train
bash scripts/setup_venv.sh
source .venv/bin/activate
python sanity_check.py
```

`setup_venv.sh` installs **CPU PyTorch**; plain `pip install torch` pulls ~2 GB
of CUDA wheels that PyBullet never uses.

`sanity_check.py` is the guard rail: it asserts obstacles actually spawn, that
the LiDAR sees them, and that a scripted controller can reach the goal. If the
scripted run does not succeed, PPO has no chance and something is broken in the
environment.

## Phase 1: navigation policy

```bash
python train_ppo.py --config configs/ppo_nav.yaml
tensorboard --logdir runs/phase1/tb
python eval_policy.py --ckpt runs/phase1/latest.zip --episodes 20
```

1.5 M steps across 8 parallel envs runs at roughly 1300 fps, about 20 minutes.
The current Phase 1 checkpoint is **19/20** success on a 20-episode eval.

## Phase 2: domain randomization

Fine-tunes Phase 1 with LiDAR noise (sigma 0.05 m), randomized spawn/obstacles,
and a lateral gust every 100 steps scaled to the drone's weight.

```bash
python train_ppo.py --config configs/ppo_nav_dr.yaml
python eval_policy.py --ckpt runs/phase2_dr/latest.zip --episodes 20 --domain-rand
python eval_policy.py --ckpt runs/phase1/latest.zip   --episodes 20 --domain-rand
```

Compare success and crash rates **with noise on** for both checkpoints and keep
the better one.

## Phase 3: ONNX export

```bash
python export_onnx.py --ckpt runs/phase2_dr/latest.zip \
                      --out ../src/drone_robot/models/drone_brain.onnx
```

Export the Phase 2 checkpoint unless Phase 1 is clearly better under `--domain-rand`.

The exported graph bakes in the action-space clip, so ONNX Runtime reproduces
`model.predict(deterministic=True)`. Export fails loudly if the two disagree by
1e-5 or more, or if the output escapes the action bounds.

## Factory corridor fine-tune

A ~30 m hallway with side walls and 12 staggered trunks, end-to-end goals.
Same 18-D observation contract. Trained from scratch on the corridor (Phase 2
forest init did not transfer). After 1.5M, an extra 750k at the same config
improved eval from 13/20 to 17/20.

```bash
python train_ppo.py --config configs/ppo_nav_factory.yaml
python train_ppo.py --config configs/ppo_nav_factory.yaml \
    --timesteps 750000 --init-from runs/phase_factory/latest.zip
python eval_policy.py --ckpt runs/phase_factory/latest.zip --episodes 20 --corridor --domain-rand --reseed
python export_onnx.py --ckpt runs/phase_factory/latest.zip \
                      --out ../src/drone_robot/models/drone_brain.onnx
```

## Notes

- `runs/` and `.venv/` are gitignored; the exported `.onnx` is committed instead.
- Installing `onnx` upgrades NumPy past 2.0, which breaks gym-pybullet-drones.
  `requirements.txt` pins `numpy<2.0`; reinstall it if you add packages.
