import math
import numpy as np


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
    target_y: float
) -> tuple[float, float]:
    """
    Compute relative target position (dx, dy) expressed in the drone's body frame.
    x points forward, y points left.
    """
    delta_x = target_x - drone_x
    delta_y = target_y - drone_y

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    body_dx = cos_yaw * delta_x + sin_yaw * delta_y
    body_dy = -sin_yaw * delta_x + cos_yaw * delta_y
    return float(body_dx), float(body_dy)


def process_laser_scan(
    ranges: list[float],
    angle_min: float,
    angle_max: float,
    angle_increment: float,
    range_min: float = 0.05,
    range_max: float = 5.0,
    num_bins: int = 16
) -> np.ndarray:
    """
    Extracts forward 180-degree field of view [-pi/2, pi/2],
    downsamples into `num_bins` equal angular sectors taking the minimum valid distance,
    and normalizes ranges to [-1.0, 1.0].
    """
    num_ranges = len(ranges)
    if num_ranges == 0:
        return np.ones(num_bins, dtype=np.float32)

    angles = angle_min + np.arange(num_ranges) * angle_increment
    ranges_arr = np.array(ranges, dtype=np.float32)

    # Filter invalid readings (inf, nan)
    ranges_arr = np.where(np.isnan(ranges_arr), range_max, ranges_arr)
    ranges_arr = np.where(np.isinf(ranges_arr), range_max, ranges_arr)
    ranges_arr = np.clip(ranges_arr, range_min, range_max)

    # Select forward 180 degrees: [-pi/2, pi/2]
    fov_mask = (angles >= -math.pi / 2.0) & (angles <= math.pi / 2.0)
    fov_angles = angles[fov_mask]
    fov_ranges = ranges_arr[fov_mask]

    if len(fov_ranges) == 0:
        return np.ones(num_bins, dtype=np.float32)

    bin_edges = np.linspace(-math.pi / 2.0, math.pi / 2.0, num_bins + 1)
    binned_ranges = np.full(num_bins, range_max, dtype=np.float32)

    for i in range(num_bins):
        in_bin = (fov_angles >= bin_edges[i]) & (fov_angles < bin_edges[i + 1])
        if np.any(in_bin):
            binned_ranges[i] = np.min(fov_ranges[in_bin])

    # Normalize from [0.0, range_max] to [-1.0, 1.0]
    normalized_ranges = 2.0 * (binned_ranges / range_max) - 1.0
    return normalized_ranges.astype(np.float32)


def assemble_observation(
    normalized_lidar: np.ndarray,
    body_dx: float,
    body_dy: float,
    max_target_dist: float = 10.0
) -> np.ndarray:
    """
    Combines 16 normalized lidar rays with body-frame (dx, dy)
    into a (1, 18) float32 array ready for ONNX inference.
    """
    norm_dx = np.clip(body_dx / max_target_dist, -1.0, 1.0)
    norm_dy = np.clip(body_dy / max_target_dist, -1.0, 1.0)

    obs = np.hstack([normalized_lidar, [norm_dx, norm_dy]]).astype(np.float32)
    return np.expand_dims(obs, axis=0)
