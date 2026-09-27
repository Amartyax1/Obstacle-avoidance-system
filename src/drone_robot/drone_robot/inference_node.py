#!/usr/bin/env python3
"""50 Hz ONNX policy: /scan + /odom -> (1, 18) observation -> /cmd_vel."""

import os
import time

import numpy as np
import onnxruntime as ort
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from drone_robot.preprocess import (
    ALTITUDE_GAIN,
    ALTITUDE_KD,
    HOVER_Z,
    MAX_VZ,
    OBS_DIM,
    assemble_observation,
    compute_relative_target_in_body_frame,
    hover_vz,
    process_laser_scan,
    quaternion_to_yaw,
)


class DroneInferenceNode(Node):
    def __init__(self):
        super().__init__('drone_inference_node')

        self.declare_parameter('model_path', '')
        self.declare_parameter('target_x', 4.0)
        self.declare_parameter('target_y', 0.0)
        self.declare_parameter('waypoint_xs', [4.0])
        self.declare_parameter('waypoint_ys', [0.0])
        self.declare_parameter('loop_waypoints', False)
        self.declare_parameter('goal_tolerance', 0.5)
        self.declare_parameter('range_max', 5.0)
        self.declare_parameter('min_vx', -0.5)
        self.declare_parameter('max_vx', 1.5)
        self.declare_parameter('max_wz', 1.0)
        self.declare_parameter('control_rate_hz', 50.0)
        self.declare_parameter('sensor_timeout_sec', 0.5)
        self.declare_parameter('hover_z', HOVER_Z)
        self.declare_parameter('altitude_gain', ALTITUDE_GAIN)
        self.declare_parameter('altitude_kd', ALTITUDE_KD)
        self.declare_parameter('max_vz', MAX_VZ)

        model_path = self.get_parameter('model_path').value
        if not model_path:
            pkg_share = get_package_share_directory('drone_robot')
            model_path = os.path.join(pkg_share, 'models', 'drone_brain.onnx')
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f'ONNX model not found at {model_path}. '
                'Export one with train/export_onnx.py.'
            )

        self.get_logger().info(f'Loading ONNX model from: {model_path}')
        self.session = ort.InferenceSession(
            model_path, providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        expected = [None, OBS_DIM]
        actual = list(self.session.get_inputs()[0].shape)
        if len(actual) != 2 or actual[1] != OBS_DIM:
            raise ValueError(f'model expects input {actual}, need {expected}')

        self.latest_scan = None
        self.last_scan_time = None
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_z = HOVER_Z
        self.drone_vz = 0.0
        self.drone_yaw = 0.0
        self.last_odom_time = None
        self.goal_reached = False
        self._wp_index = 0
        self._infer_times = []
        self._last_cmd = (0.0, 0.0, 0.0)

        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, qos_profile_sensor_data
        )
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        rate_hz = float(self.get_parameter('control_rate_hz').value)
        self.timer = self.create_timer(1.0 / rate_hz, self.control_loop)
        self.stats_timer = self.create_timer(5.0, self.log_stats)
        self.get_logger().info(f'Inference node ready at {rate_hz:.0f} Hz.')

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg
        self.last_scan_time = self.get_clock().now()

    def odom_callback(self, msg: Odometry):
        self.drone_x = msg.pose.pose.position.x
        self.drone_y = msg.pose.pose.position.y
        self.drone_z = msg.pose.pose.position.z
        self.drone_vz = msg.twist.twist.linear.z
        q = msg.pose.pose.orientation
        self.drone_yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self.last_odom_time = self.get_clock().now()

    def _altitude_cmd(self) -> float:
        return hover_vz(
            self.drone_z,
            self.drone_vz,
            float(self.get_parameter('hover_z').value),
            float(self.get_parameter('altitude_gain').value),
            float(self.get_parameter('altitude_kd').value),
            float(self.get_parameter('max_vz').value),
        )

    def _waypoints(self):
        xs = [float(v) for v in self.get_parameter('waypoint_xs').value]
        ys = [float(v) for v in self.get_parameter('waypoint_ys').value]
        if not xs or len(xs) != len(ys):
            return (
                [float(self.get_parameter('target_x').value)],
                [float(self.get_parameter('target_y').value)],
            )
        return xs, ys

    def _active_goal(self):
        xs, ys = self._waypoints()
        index = min(self._wp_index, len(xs) - 1)
        return xs[index], ys[index], len(xs)

    def _advance_waypoint(self, wp_count: int) -> None:
        if wp_count <= 1:
            return
        nxt = self._wp_index + 1
        if nxt >= wp_count:
            if not bool(self.get_parameter('loop_waypoints').value):
                return
            nxt = 0
        self._wp_index = nxt
        self.goal_reached = False
        xs, ys = self._waypoints()
        self.get_logger().info(f'Next waypoint ({xs[nxt]:.1f}, {ys[nxt]:.1f}).')

    def _stale(self, stamp, timeout_sec: float) -> bool:
        if stamp is None:
            return True
        age = (self.get_clock().now() - stamp).nanoseconds * 1e-9
        return age > timeout_sec

    def control_loop(self):
        timeout = float(self.get_parameter('sensor_timeout_sec').value)
        if self._stale(self.last_odom_time, timeout) or self._stale(
            self.last_scan_time, timeout
        ):
            self.cmd_pub.publish(Twist())
            return

        target_x, target_y, wp_count = self._active_goal()
        range_max = float(self.get_parameter('range_max').value)
        tolerance = float(self.get_parameter('goal_tolerance').value)

        body_dx, body_dy = compute_relative_target_in_body_frame(
            self.drone_x, self.drone_y, self.drone_yaw, target_x, target_y
        )

        if np.hypot(body_dx, body_dy) < tolerance:
            hold = Twist()
            hold.linear.z = self._altitude_cmd()
            self.cmd_pub.publish(hold)
            if not self.goal_reached:
                self.goal_reached = True
                self.get_logger().info(
                    f'Waypoint {self._wp_index + 1}/{max(wp_count, 1)} '
                    f'({target_x:.1f}, {target_y:.1f}) reached at '
                    f'({self.drone_x:.2f}, {self.drone_y:.2f}, {self.drone_z:.2f}).'
                )
                self._advance_waypoint(wp_count)
            return
        self.goal_reached = False

        scan = self.latest_scan
        normalized_lidar = process_laser_scan(
            ranges=scan.ranges,
            angle_min=scan.angle_min,
            angle_increment=scan.angle_increment,
            range_max=range_max,
        )
        obs = assemble_observation(
            normalized_lidar, body_dx, body_dy, max_target_dist=range_max
        )

        start = time.perf_counter()
        action = self.session.run([self.output_name], {self.input_name: obs})[0][0]
        self._infer_times.append((time.perf_counter() - start) * 1e3)

        min_vx = float(self.get_parameter('min_vx').value)
        max_vx = float(self.get_parameter('max_vx').value)
        max_wz = float(self.get_parameter('max_wz').value)

        cmd = Twist()
        cmd.linear.x = float(np.clip(action[0], min_vx, max_vx))
        cmd.linear.z = self._altitude_cmd()
        cmd.angular.z = float(np.clip(action[1], -max_wz, max_wz))
        self._last_cmd = (cmd.linear.x, cmd.linear.z, cmd.angular.z)
        self.cmd_pub.publish(cmd)

    def log_stats(self):
        if not self._infer_times:
            return
        times = np.asarray(self._infer_times)
        self._infer_times = []
        vx, vz, wz = self._last_cmd
        self.get_logger().info(
            f'inference {times.mean():.2f} ms mean / {times.max():.2f} ms max '
            f'over {times.size} frames  pose=({self.drone_x:.2f}, {self.drone_y:.2f}, '
            f'{self.drone_z:.2f})  vx={vx:+.2f} vz={vz:+.2f} wz={wz:+.2f}'
        )


def main(args=None):
    rclpy.init(args=args)
    node = DroneInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
