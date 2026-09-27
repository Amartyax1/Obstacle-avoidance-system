"""
LaserScan/Odometry to (1, 18) observation, matching train/envs/lidar_nav_env.py.

The training env casts 16 rays over a 180 deg forward fan, clips to 5 m, maps to
[-1, 1], and appends the body-frame goal vector scaled by the same 5 m. Any change
here has to be mirrored there or the policy sees out-of-distribution input.
"""

import math

import numpy as np

N_RAYS = 16
OBS_DIM = 18
MAX_RANGE = 5.0
HOVER_Z = 1.0
ALTITUDE_GAIN = 1.8
ALTITUDE_KD = 1.2
MAX_VZ = 1.0


def hover_vz(
    z: float,
    z_dot: float = 0.0,
    hover_z: float = HOVER_Z,
    gain: float = ALTITUDE_GAIN,
    kd: float = ALTITUDE_KD,
    limit: float = MAX_VZ,
) -> float:
    """PD altitude hold; P-term matches LidarNavEnv, D-term damps Gazebo overshoot."""
    return float(np.clip(gain * (hover_z - z) - kd * z_dot, -limit, limit))


def quaternion_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw angle (radians) from quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def compute_relative_target_in_body_frame(
    drone_x: float,
    drone_y: float,
    yaw: float,
    target_x: float,
    target_y: float,
) -> tuple[float, float]:
    """Relative target (dx, dy) in the drone body frame: x forward, y left."""
    delta_x = target_x - drone_x
    delta_y = target_y - drone_y

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    body_dx = cos_yaw * delta_x + sin_yaw * delta_y
    body_dy = -sin_yaw * delta_x + cos_yaw * delta_y
    return float(body_dx), float(body_dy)


def process_laser_scan(
    ranges,
    angle_min: float,
    angle_increment: float,
    range_min: float = 0.05,
    range_max: float = MAX_RANGE,
    num_bins: int = N_RAYS,
) -> np.ndarray:
    """
    Downsample the forward 180 deg fan into `num_bins` sectors, normalized to [-1, 1].

    NaN/Inf mean "nothing detected" and become max range. Each bin keeps the
    closest return so obstacles are never averaged away.
    """
    ranges_arr = np.asarray(ranges, dtype=np.float32)
    if ranges_arr.size == 0:
        return np.ones(num_bins, dtype=np.float32)

    angles = angle_min + np.arange(ranges_arr.size, dtype=np.float32) * angle_increment

    valid = np.isfinite(ranges_arr)
    ranges_arr = np.where(valid, ranges_arr, range_max)
    ranges_arr = np.clip(ranges_arr, range_min, range_max)

    half_fov = math.pi / 2.0
    fov_mask = (angles >= -half_fov) & (angles <= half_fov)
    if not np.any(fov_mask):
        return np.ones(num_bins, dtype=np.float32)
    fov_angles = angles[fov_mask]
    fov_ranges = ranges_arr[fov_mask]

    # Bin index per ray; the final edge is inclusive so +90 deg is not dropped.
    bin_width = (2.0 * half_fov) / num_bins
    idx = np.clip(((fov_angles + half_fov) / bin_width).astype(np.int32), 0, num_bins - 1)

    binned = np.full(num_bins, range_max, dtype=np.float32)
    np.minimum.at(binned, idx, fov_ranges)

    return (2.0 * (binned / range_max) - 1.0).astype(np.float32)


def assemble_observation(
    normalized_lidar: np.ndarray,
    body_dx: float,
    body_dy: float,
    max_target_dist: float = MAX_RANGE,
) -> np.ndarray:
    """Combine 16 normalized rays with body-frame (dx, dy) into (1, 18) float32."""
    norm_dx = np.clip(body_dx / max_target_dist, -1.0, 1.0)
    norm_dy = np.clip(body_dy / max_target_dist, -1.0, 1.0)

    obs = np.hstack([normalized_lidar, [norm_dx, norm_dy]]).astype(np.float32)
    return np.expand_dims(obs, axis=0)
