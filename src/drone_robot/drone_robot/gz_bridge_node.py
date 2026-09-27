#!/usr/bin/env python3
"""
Bridge Harmonic gz.transport topics to ROS 2.

Humble ros_gz_bridge 0.244 is built against a different gz-msgs ABI than
gz-sim8 on this machine (Unknown message type [8]/[9]), so the stock
parameter_bridge never delivers /scan, /odom, or /clock.
"""

from queue import SimpleQueue
from threading import Lock

from gz.msgs10.clock_pb2 import Clock as GzClock
from gz.msgs10.laserscan_pb2 import LaserScan as GzLaserScan
from gz.msgs10.odometry_pb2 import Odometry as GzOdometry
from gz.msgs10.twist_pb2 import Twist as GzTwist
from gz.transport13 import Node as GzNode
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import LaserScan

import rclpy

SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)

CLOCK_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def _stamp_msg(ros_stamp, gz_header):
    ros_stamp.sec = int(gz_header.stamp.sec)
    ros_stamp.nanosec = int(gz_header.stamp.nsec)


class HarmonicBridge(Node):
    def __init__(self):
        super().__init__('harmonic_gz_bridge')
        self.gz = GzNode()

        self.scan_pub = self.create_publisher(LaserScan, '/scan', SENSOR_QOS)
        self.odom_pub = self.create_publisher(Odometry, '/odom', SENSOR_QOS)
        self.clock_pub = self.create_publisher(Clock, '/clock', CLOCK_QOS)
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd, 10)

        self._twist_pub = self.gz.advertise(
            '/drone_robot/gazebo/command/twist', GzTwist
        )

        self._queue = SimpleQueue()
        self._twist_lock = Lock()
        self.create_timer(0.01, self._flush_queue)

        self.gz.subscribe(GzLaserScan, '/scan', self._on_scan)
        self.gz.subscribe(GzOdometry, '/odom', self._on_odom)
        self.gz.subscribe(GzClock, '/clock', self._on_clock)

        self.get_logger().info(
            'Harmonic gz.transport bridge up: /scan /odom /clock <-> /cmd_vel'
        )

    def _on_scan(self, msg: GzLaserScan):
        out = LaserScan()
        _stamp_msg(out.header.stamp, msg.header)
        out.header.frame_id = msg.frame or 'lidar_link'
        out.angle_min = float(msg.angle_min)
        out.angle_max = float(msg.angle_max)
        out.angle_increment = float(msg.angle_step)
        out.range_min = float(msg.range_min)
        out.range_max = float(msg.range_max)
        out.ranges = [float(r) for r in msg.ranges]
        if msg.intensities:
            out.intensities = [float(i) for i in msg.intensities]
        self._queue.put(('scan', out))

    def _on_odom(self, msg: GzOdometry):
        out = Odometry()
        _stamp_msg(out.header.stamp, msg.header)
        out.header.frame_id = 'odom'
        out.child_frame_id = 'base_link'
        out.pose.pose.position.x = float(msg.pose.position.x)
        out.pose.pose.position.y = float(msg.pose.position.y)
        out.pose.pose.position.z = float(msg.pose.position.z)
        out.pose.pose.orientation.x = float(msg.pose.orientation.x)
        out.pose.pose.orientation.y = float(msg.pose.orientation.y)
        out.pose.pose.orientation.z = float(msg.pose.orientation.z)
        out.pose.pose.orientation.w = float(msg.pose.orientation.w)
        out.twist.twist.linear.x = float(msg.twist.linear.x)
        out.twist.twist.linear.y = float(msg.twist.linear.y)
        out.twist.twist.linear.z = float(msg.twist.linear.z)
        out.twist.twist.angular.x = float(msg.twist.angular.x)
        out.twist.twist.angular.y = float(msg.twist.angular.y)
        out.twist.twist.angular.z = float(msg.twist.angular.z)
        self._queue.put(('odom', out))

    def _on_clock(self, msg: GzClock):
        out = Clock()
        out.clock.sec = int(msg.sim.sec)
        out.clock.nanosec = int(msg.sim.nsec)
        self._queue.put(('clock', out))

    def _flush_queue(self):
        while not self._queue.empty():
            kind, msg = self._queue.get_nowait()
            if kind == 'scan':
                self.scan_pub.publish(msg)
            elif kind == 'odom':
                self.odom_pub.publish(msg)
            elif kind == 'clock':
                self.clock_pub.publish(msg)

    def _on_cmd(self, msg: Twist):
        gz = GzTwist()
        gz.linear.x = float(msg.linear.x)
        gz.linear.y = float(msg.linear.y)
        gz.linear.z = float(msg.linear.z)
        gz.angular.x = float(msg.angular.x)
        gz.angular.y = float(msg.angular.y)
        gz.angular.z = float(msg.angular.z)
        with self._twist_lock:
            self._twist_pub.publish(gz)


def main(args=None):
    rclpy.init(args=args)
    node = HarmonicBridge()
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
