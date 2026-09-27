"""Preprocessing must reproduce the training observation exactly."""

import math

import numpy as np
import pytest

from drone_robot.preprocess import (
    HOVER_Z,
    MAX_RANGE,
    N_RAYS,
    OBS_DIM,
    assemble_observation,
    compute_relative_target_in_body_frame,
    hover_vz,
    process_laser_scan,
    quaternion_to_yaw,
)


def make_scan(values, n=180):
    """Full 180 deg forward fan with `n` rays."""
    angle_min = -math.pi / 2.0
    angle_increment = math.pi / (n - 1)
    ranges = np.full(n, values, dtype=np.float32) if np.isscalar(values) else values
    return list(ranges), angle_min, angle_increment


def test_yaw_from_quaternion():
    assert quaternion_to_yaw(0.0, 0.0, 0.0, 1.0) == pytest.approx(0.0)
    # 90 deg about z
    assert quaternion_to_yaw(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)) == pytest.approx(
        math.pi / 2.0
    )


def test_body_frame_target_rotates_with_yaw():
    # Target straight ahead in world +x, drone facing +x.
    dx, dy = compute_relative_target_in_body_frame(0.0, 0.0, 0.0, 3.0, 0.0)
    assert dx == pytest.approx(3.0)
    assert dy == pytest.approx(0.0)

    # Same target, drone rotated 90 deg left -> target is now on the right.
    dx, dy = compute_relative_target_in_body_frame(0.0, 0.0, math.pi / 2.0, 3.0, 0.0)
    assert dx == pytest.approx(0.0, abs=1e-6)
    assert dy == pytest.approx(-3.0)


def test_clear_scan_maps_to_upper_bound():
    ranges, amin, ainc = make_scan(MAX_RANGE)
    out = process_laser_scan(ranges, amin, ainc)
    assert out.shape == (N_RAYS,)
    assert np.allclose(out, 1.0)


def test_nan_and_inf_treated_as_max_range():
    ranges, amin, ainc = make_scan(MAX_RANGE)
    ranges[10] = float('nan')
    ranges[20] = float('inf')
    ranges[30] = float('-inf')
    out = process_laser_scan(ranges, amin, ainc)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 1.0)


def test_bin_keeps_closest_return():
    n = 180
    ranges, amin, ainc = make_scan(MAX_RANGE, n=n)
    # One close return in the first bin must dominate that bin.
    ranges[2] = 1.0
    out = process_laser_scan(ranges, amin, ainc)
    assert out[0] == pytest.approx(2.0 * (1.0 / MAX_RANGE) - 1.0)
    assert np.allclose(out[1:], 1.0)


def test_last_bin_includes_final_ray():
    n = 180
    ranges, amin, ainc = make_scan(MAX_RANGE, n=n)
    ranges[-1] = 0.5
    out = process_laser_scan(ranges, amin, ainc)
    assert out[-1] == pytest.approx(2.0 * (0.5 / MAX_RANGE) - 1.0)


def test_wider_scan_is_cropped_to_forward_fan():
    # 360 deg scan: readings behind the drone must be ignored.
    n = 360
    angle_min = -math.pi
    angle_increment = 2.0 * math.pi / (n - 1)
    ranges = np.full(n, MAX_RANGE, dtype=np.float32)
    ranges[0] = 0.1  # directly behind
    out = process_laser_scan(list(ranges), angle_min, angle_increment)
    assert np.allclose(out, 1.0)


def test_empty_scan_is_safe():
    out = process_laser_scan([], -math.pi / 2.0, 0.01)
    assert out.shape == (N_RAYS,)
    assert np.allclose(out, 1.0)


def test_observation_shape_and_bounds():
    lidar = np.zeros(N_RAYS, dtype=np.float32)
    obs = assemble_observation(lidar, 100.0, -100.0)
    assert obs.shape == (1, OBS_DIM)
    assert obs.dtype == np.float32
    assert obs[0, -2] == pytest.approx(1.0)
    assert obs[0, -1] == pytest.approx(-1.0)


def test_goal_scaling_matches_training_range():
    lidar = np.zeros(N_RAYS, dtype=np.float32)
    obs = assemble_observation(lidar, MAX_RANGE / 2.0, 0.0)
    # Training divides the body-frame goal by MAX_RANGE, not by 10.
    assert obs[0, -2] == pytest.approx(0.5)


def test_hover_vz_matches_training_law():
    assert hover_vz(HOVER_Z) == pytest.approx(0.0)
    assert hover_vz(0.5) == pytest.approx(0.9)
    assert hover_vz(0.0) == pytest.approx(1.0)
    assert hover_vz(2.0) == pytest.approx(-1.0)
    # D-term opposes climb so Gazebo does not overshoot the 2 m tree tops.
    assert hover_vz(HOVER_Z, z_dot=0.5) == pytest.approx(-0.6)
