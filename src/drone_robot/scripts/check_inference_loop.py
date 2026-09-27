#!/usr/bin/env python3
"""
Closed-loop smoke test for the inference node without Gazebo.

Publishes synthetic /scan and /odom, runs the real DroneInferenceNode against the
exported ONNX model, and reports the achieved /cmd_vel rate.

    python3 src/drone_robot/scripts/check_inference_loop.py
"""

import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from drone_robot.inference_node import DroneInferenceNode

DURATION_SEC = 5.0
SCAN_RAYS = 180


class FakeSim(Node):
    """Stands in for Gazebo: constant obstacle field, drone drifting forward."""

    def __init__(self):
        super().__init__('fake_sim')
        self.scan_pub = self.create_publisher(
            LaserScan, '/scan', qos_profile_sensor_data
        )
        self.odom_pub = self.create_publisher(
            Odometry, '/odom', qos_profile_sensor_data
        )
        self.cmd_count = 0
        self.last_cmd = None
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd, 10)
        self.create_timer(1.0 / 30.0, self.publish_scan)
        self.create_timer(1.0 / 50.0, self.publish_odom)
        self.x = 0.0

    def on_cmd(self, msg: Twist):
        self.cmd_count += 1
        self.last_cmd = msg

    def publish_scan(self):
        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'lidar_link'
        msg.angle_min = -math.pi / 2.0
        msg.angle_max = math.pi / 2.0
        msg.angle_increment = math.pi / (SCAN_RAYS - 1)
        msg.range_min = 0.05
        msg.range_max = 5.0
        ranges = [5.0] * SCAN_RAYS
        ranges[90] = 1.2                 # obstacle dead ahead
        ranges[45] = float('inf')        # nothing detected
        ranges[120] = float('nan')       # dropout
        msg.ranges = ranges
        self.scan_pub.publish(msg)

    def publish_odom(self):
        self.x += 0.005
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.pose.pose.position.x = self.x
        msg.pose.pose.position.z = 1.0
        msg.pose.pose.orientation.w = 1.0
        self.odom_pub.publish(msg)


def main():
    rclpy.init()
    sim = FakeSim()
    try:
        node = DroneInferenceNode()
    except FileNotFoundError as exc:
        print(f'FAIL: {exc}')
        rclpy.shutdown()
        return 1

    executor = SingleThreadedExecutor()
    executor.add_node(sim)
    executor.add_node(node)

    start = time.time()
    while time.time() - start < DURATION_SEC:
        executor.spin_once(timeout_sec=0.01)
    elapsed = time.time() - start

    rate = sim.cmd_count / elapsed
    target = float(node.get_parameter('control_rate_hz').value)
    print(f'/cmd_vel messages: {sim.cmd_count} in {elapsed:.2f}s -> {rate:.1f} Hz '
          f'(target {target:.0f} Hz)')
    if sim.last_cmd is not None:
        print(f'last command: vx={sim.last_cmd.linear.x:+.3f} '
              f'vz={sim.last_cmd.linear.z:+.3f} '
              f'wz={sim.last_cmd.angular.z:+.3f}')

    node.destroy_node()
    sim.destroy_node()
    rclpy.shutdown()

    if sim.cmd_count == 0:
        print('FAIL: node never published a command')
        return 1
    if rate < 0.8 * target:
        print(f'FAIL: loop rate {rate:.1f} Hz below 80% of {target:.0f} Hz')
        return 1
    print('PASS: closed loop running at target rate')
    return 0


if __name__ == '__main__':
    sys.exit(main())
