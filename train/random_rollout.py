#!/usr/bin/env python3
"""Sanity rollout: random actions should crash or timeout, not hang."""

from __future__ import annotations

import numpy as np

from envs.lidar_nav_env import LidarNavEnv, OBS_DIM


def main():
    env = LidarNavEnv(gui=False, domain_rand=False)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_DIM,), obs.shape
    assert env.action_space.shape == (2,)
    print("reset ok", obs[:4], "goal_dist", info["goal_distance"])
    for i in range(120):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print(f"episode ended at step {i} reward={reward:.2f} info={info}")
            obs, info = env.reset()
            break
    else:
        print("ran 120 steps without terminal; last reward", reward, "dist", info["goal_distance"])
    env.close()
    print("random rollout ok")


if __name__ == "__main__":
    main()
