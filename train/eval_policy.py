#!/usr/bin/env python3
"""Evaluate a PPO checkpoint. Run from train/: python eval_policy.py --ckpt runs/phase1/latest.zip"""

from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from envs.lidar_nav_env import LidarNavEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--domain-rand", action="store_true")
    parser.add_argument(
        "--warehouse",
        action="store_true",
        help="procedural warehouse URDF (same boxes as Gazebo)",
    )
    parser.add_argument(
        "--corridor",
        action="store_true",
        help="30 m walled hallway with end goals (factory proxy)",
    )
    parser.add_argument(
        "--explore",
        action="store_true",
        help="enclosed arena, live occupancy A* lookahead (Stage 3)",
    )
    parser.add_argument(
        "--reseed",
        action="store_true",
        help="reset(seed=episode_index) each episode (harder, less like training)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    env_kwargs = {"gui": args.gui, "domain_rand": args.domain_rand}
    if args.warehouse:
        env_kwargs.update(
            layout="warehouse",
            n_obstacles=0,
            world_size=38.0,
            episode_len_sec=40.0,
        )
    elif args.corridor:
        env_kwargs.update(
            layout="corridor",
            n_obstacles=12,
            world_size=30.0,
            episode_len_sec=40.0,
        )
    elif args.explore:
        env_kwargs.update(
            layout="explore",
            n_obstacles=10,
            world_size=20.0,
            episode_len_sec=40.0,
        )
    env = LidarNavEnv(**env_kwargs)
    model = PPO.load(args.ckpt, device="cpu")
    successes = 0
    crashes = 0
    timeouts = 0
    returns = []
    env.reset(seed=0)
    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep if args.reseed else None)
        done = False
        ep_ret = 0.0
        last_info = info
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, last_info = env.step(action)
            ep_ret += float(reward)
            done = terminated or truncated
        returns.append(ep_ret)
        if last_info.get("success"):
            successes += 1
        elif last_info.get("collision"):
            crashes += 1
        else:
            timeouts += 1
        print(
            f"ep {ep:02d} return={ep_ret:7.2f} "
            f"dist={last_info.get('goal_distance', float('nan')):.2f} "
            f"success={last_info.get('success')} collision={last_info.get('collision')}"
        )
    env.close()
    n = args.episodes
    print("---")
    print(f"mean return {np.mean(returns):.2f}")
    print(f"success {successes}/{n}  crash {crashes}/{n}  timeout {timeouts}/{n}")


if __name__ == "__main__":
    main()
