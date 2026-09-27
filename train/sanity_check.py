#!/usr/bin/env python3
"""Checks the MDP is wired up and solvable before spending time on PPO.

1. obstacles really exist in the PyBullet world
2. a random policy runs without crashing the sim
3. a scripted go-to-goal controller reaches the target often
"""

from __future__ import annotations

import argparse

import numpy as np

from envs.lidar_nav_env import GOAL_RADIUS, MAX_RANGE, N_RAYS, OBS_DIM, LidarNavEnv


def scripted_action(obs: np.ndarray) -> np.ndarray:
    """Turn toward the goal, slow down when the forward sector is blocked."""
    rays = (obs[:N_RAYS] + 1.0) / 2.0 * MAX_RANGE
    dx, dy = obs[N_RAYS] * MAX_RANGE, obs[N_RAYS + 1] * MAX_RANGE

    bearing = np.arctan2(dy, dx)
    wz = float(np.clip(1.5 * bearing, -1.0, 1.0))

    front = rays[N_RAYS // 2 - 3 : N_RAYS // 2 + 3]
    clearance = float(np.min(front))
    if clearance < 1.0:
        # Steer toward the side with more room instead of pushing through.
        left = float(np.mean(rays[N_RAYS // 2 :]))
        right = float(np.mean(rays[: N_RAYS // 2]))
        wz = 1.0 if left > right else -1.0
        vx = 0.3
    else:
        vx = float(np.clip(0.6 * np.hypot(dx, dy), 0.3, 1.5))
    return np.array([vx, wz], dtype=np.float32)


def run_episodes(env, policy, episodes: int, label: str) -> float:
    successes = 0
    collisions = 0
    for ep in range(episodes):
        obs, info = env.reset(seed=1000 + ep)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            done = terminated or truncated
        successes += bool(info["success"])
        collisions += bool(info["collision"])
    rate = successes / episodes
    print(f"{label}: success {successes}/{episodes}  crash {collisions}/{episodes}")
    return rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    env = LidarNavEnv(gui=False, domain_rand=False)
    obs, info = env.reset(seed=0)

    assert obs.shape == (OBS_DIM,), obs.shape
    assert env.action_space.shape == (2,)
    assert len(env.obstacle_ids) == env.n_obstacles, (
        f"expected {env.n_obstacles} obstacles, got {len(env.obstacle_ids)}"
    )
    print(f"obstacles spawned: {len(env.obstacle_ids)}")
    print(f"goal distance at reset: {info['goal_distance']:.2f} m")
    print(f"min lidar range at reset: {info['min_range']:.2f} m")
    assert info["min_range"] < MAX_RANGE, "lidar never sees an obstacle"

    run_episodes(env, lambda _: env.action_space.sample(), args.episodes, "random  ")
    rate = run_episodes(env, scripted_action, args.episodes, "scripted")
    env.close()

    assert rate > 0.0, "scripted controller never reached the goal; MDP may be broken"
    print(f"sanity ok (goal radius {GOAL_RADIUS} m)")


if __name__ == "__main__":
    main()
