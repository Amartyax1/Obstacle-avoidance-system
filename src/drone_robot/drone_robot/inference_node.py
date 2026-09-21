#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
import numpy as np
import onnxruntime as ort
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from drone_robot.preprocess import (
    quaternion_to_yaw,
    compute_relative_target_in_body_frame,
    process_laser_scan,
    assemble_observation
)


class DroneInferenceNode(Node):
    def __init__(self):
        super().__init__('drone_inference_node')

        # Parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('target_x', 5.0)
        self.declare_parameter('target_y', 0.0)
        self.declare_parameter('max_vx', 1.5)
        self.declare_parameter('min_vx', -0.5)
        self.declare_parameter('max_wz', 1.0)
        self.declare_parameter('range_max', 5.0)
        self.declare_parameter('control_rate_hz', 50.0)

        model_path = self.get_parameter('model_path').value
        if not model_path:
            pkg_share = get_package_share_directory('drone_robot')
            model_path = os.path.join(pkg_share, 'models', 'drone_brain.onnx')

        self.get_logger().info(f'Loading ONNX model from: {model_path}')
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # State storage
        self.latest_scan = None
        self.drone_x = 0.0
        self.drone_y = 0.0
        self.drone_yaw = 0.0
        self.has_odom = False

        # Subscriptions
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer loop for inference (50Hz)
        rate_hz = self.get_parameter('control_rate_hz').value
        self.timer = self.create_timer(1.0 / rate_hz, self.control_loop)
        self.get_logger().info('Drone ONNX Inference Node initialized.')

    def scan_callback(self, msg: LaserScan):
        self.latest_scan = msg

    def odom_callback(self, msg: Odometry):
        self.drone_x = msg.pose.pose.position.x
        self.drone_y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        self.drone_yaw = quaternion_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )
        self.has_odom = True

    def control_loop(self):
        if not self.has_odom or self.latest_scan is None:
            return

        target_x = self.get_parameter('target_x').value
        target_y = self.get_parameter('target_y').value
        range_max = self.get_parameter('range_max').value

        # Calculate relative position in drone frame
        body_dx, body_dy = compute_relative_target_in_body_frame(
            self.drone_x,
            self.drone_y,
            self.drone_yaw,
            target_x,
            target_y
        )

        dist_to_target = (body_dx ** 2 + body_dy ** 2) ** 0.5
        if dist_to_target < 0.3:
            # Stop upon reaching target waypoint
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            self.get_logger().info('Target waypoint reached! Hovering.')
            return

        # Preprocess scan
        normalized_lidar = process_laser_scan(
            ranges=self.latest_scan.ranges,
            angle_min=self.latest_scan.angle_min,
            angle_max=self.latest_scan.angle_max,
            angle_increment=self.latest_scan.angle_increment,
            range_max=range_max,
            num_bins=16
        )

        obs = assemble_observation(normalized_lidar, body_dx, body_dy)

        # Run ONNX inference
        outputs = self.session.run([self.output_name], {self.input_name: obs})
        action = outputs[0][0]  # shape: (2,) [vx, wz]

        # Action mapping
        min_vx = self.get_parameter('min_vx').value
        max_vx = self.get_parameter('max_vx').value
        max_wz = self.get_parameter('max_wz').value

        vx = float(np.clip(action[0], min_vx, max_vx))
        wz = float(np.clip(action[1], -max_wz, max_wz))

        # Publish Twist message
        cmd = Twist()
        cmd.linear.x = vx
        cmd.angular.z = wz
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = DroneInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
