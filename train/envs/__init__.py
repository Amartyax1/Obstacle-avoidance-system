"""Nav training envs. LidarNavEnv is imported lazily so occupancy tests skip PyBullet."""

from __future__ import annotations

__all__ = ["LidarNavEnv"]


def __getattr__(name: str):
    if name == "LidarNavEnv":
        from .lidar_nav_env import LidarNavEnv

        return LidarNavEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
