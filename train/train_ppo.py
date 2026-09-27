#!/usr/bin/env python3
"""Train PPO on LidarNavEnv. Run from train/: python train_ppo.py --config configs/ppo_nav.yaml"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from envs.lidar_nav_env import LidarNavEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_nav.yaml")
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--init-from", default=None, help="checkpoint to fine-tune")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    n_envs = args.n_envs if args.n_envs is not None else int(cfg["n_envs"])
    timesteps = args.timesteps if args.timesteps is not None else int(cfg["n_timesteps"])
    domain_rand = bool(cfg.get("domain_rand", False))
    log_dir = Path(cfg.get("log_dir", "runs/phase1"))
    log_dir.mkdir(parents=True, exist_ok=True)
    env_kwargs = {
        "gui": False,
        "domain_rand": domain_rand,
        "layout": str(cfg.get("layout", "forest")),
        "n_obstacles": int(cfg.get("n_obstacles", 8)),
        "world_size": float(cfg.get("world_size", 8.0)),
        "episode_len_sec": float(cfg.get("episode_len_sec", 20.0)),
    }

    vec_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    env = make_vec_env(
        LidarNavEnv,
        n_envs=n_envs,
        seed=int(cfg.get("seed", 0)),
        vec_env_cls=vec_cls,
        env_kwargs=env_kwargs,
    )

    init_from = args.init_from or cfg.get("init_from")
    if init_from:
        print(f"fine-tuning from {init_from}")
        model = PPO.load(init_from, env=env, device="cpu",
                         tensorboard_log=str(log_dir / "tb"))
        model.learning_rate = float(cfg.get("learning_rate", 1e-4))
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=int(cfg.get("seed", 0)),
            learning_rate=float(cfg.get("learning_rate", 3e-4)),
            n_steps=int(cfg.get("n_steps", 1024)),
            batch_size=int(cfg.get("batch_size", 256)),
            gamma=float(cfg.get("gamma", 0.99)),
            tensorboard_log=str(log_dir / "tb"),
            policy_kwargs={"net_arch": list(cfg.get("net_arch", [64, 64]))},
        )
    ckpt = CheckpointCallback(
        save_freq=max(int(cfg.get("checkpoint_every", 50_000)) // n_envs, 1),
        save_path=str(log_dir / "ckpts"),
        name_prefix="ppo_lidar",
    )
    model.learn(total_timesteps=timesteps, callback=ckpt)
    out = log_dir / "latest.zip"
    model.save(str(out))
    env.close()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
