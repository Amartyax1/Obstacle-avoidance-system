#!/usr/bin/env python3
"""A* GPS co-pilot: live occupancy from /scan, always-on 2 m lookahead. Never /cmd_vel."""

from __future__ import annotations

import math
import os

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32
from visualization_msgs.msg import Marker

from drone_robot.grid_astar import astar, lookahead_on_path
from drone_robot.online_occupancy import (
    GRID_RES,
    INFLATE_M,
    carve_rays,
    make_grid,
)
from drone_robot.preprocess import (
    MAX_RANGE,
    N_RAYS,
    binned_lidar_ranges,
    quaternion_to_yaw,
)


class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')
        self.declare_parameter('online_map', True)
        self.declare_parameter('grid_path', '')
        self.declare_parameter('xmin', -12.0)
        self.declare_parameter('xmax', 12.0)
        self.declare_parameter('ymin', -12.0)
        self.declare_parameter('ymax', 12.0)
        self.declare_parameter('grid_res', GRID_RES)
        self.declare_parameter('inflate_m', INFLATE_M)
        self.declare_parameter('goal_x', 8.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('lookahead_m', 2.0)
        self.declare_parameter('range_max', MAX_RANGE)

        online = bool(self.get_parameter('online_map').value)
        if online:
            self.occ, self.xs, self.ys = make_grid(
                float(self.get_parameter('xmin').value),
                float(self.get_parameter('xmax').value),
                float(self.get_parameter('ymin').value),
                float(self.get_parameter('ymax').value),
                float(self.get_parameter('grid_res').value),
            )
            self.global_goal = (
                float(self.get_parameter('goal_x').value),
                float(self.get_parameter('goal_y').value),
            )
            self.x = 0.0
            self.y = 0.0
            self.get_logger().info(
                'A* planner: optimistic live grid '
                f'({self.xs[0]:.1f}..{self.xs[-1]:.1f}, '
                f'{self.ys[0]:.1f}..{self.ys[-1]:.1f}) '
                f'goal=({self.global_goal[0]:.1f}, {self.global_goal[1]:.1f})'
            )
        else:
            grid_path = self.get_parameter('grid_path').value
            if not grid_path:
                share = get_package_share_directory('drone_robot')
                grid_path = os.path.join(share, 'config', 'warehouse_grid.npz')
            data = np.load(grid_path, allow_pickle=True)
            self.occ = data['occupancy']
            self.xs = data['xs']
            self.ys = data['ys']
            stored_goal = np.asarray(data['goal'], dtype=np.float64)
            self.global_goal = (
                float(self.get_parameter('goal_x').value),
                float(self.get_parameter('goal_y').value),
            )
            spawn = np.asarray(data['spawn'], dtype=np.float64)
            self.x = float(spawn[0])
            self.y = float(spawn[1])
            self.get_logger().info(f'A* planner loaded cheat grid {grid_path}')

        self.yaw = 0.0
        self.interventions = 0
        self.path: list[tuple[float, float]] = []
        self._have_odom = False
        self._have_scan = False
        self._blocked = False

        self.look_pub = self.create_publisher(PoseStamped, '/lookahead_goal', 10)
        self.path_pub = self.create_publisher(Path, '/global_path', 10)
        self.count_pub = self.create_publisher(Int32, '/planner/interventions', 10)
        self.marker_pub = self.create_publisher(Marker, '/lookahead_marker', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/planner/occupancy', 1)

        self.create_subscription(Odometry, '/odom', self._on_odom, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self._on_scan, qos_profile_sensor_data)
        self.create_timer(0.1, self._tick)

    def _now(self):
        return self.get_clock().now()

    def _on_odom(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.yaw = quaternion_to_yaw(q.x, q.y, q.z, q.w)
        self._have_odom = True

    def _on_scan(self, msg: LaserScan):
        if not self._have_odom:
            return
        range_max = float(self.get_parameter('range_max').value)
        ranges = binned_lidar_ranges(
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            range_max=range_max,
        )
        headings = self.yaw + np.linspace(-math.pi / 2.0, math.pi / 2.0, N_RAYS)
        carve_rays(
            self.occ,
            self.xs,
            self.ys,
            (self.x, self.y),
            ranges,
            headings,
            range_max,
            inflate_m=float(self.get_parameter('inflate_m').value),
        )
        self._have_scan = True

    def _plan(self):
        return astar(
            self.occ,
            self.xs,
            self.ys,
            (self.x, self.y),
            self.global_goal,
        )

    def _publish_path(self, path):
        msg = Path()
        msg.header.stamp = self._now().to_msg()
        msg.header.frame_id = 'odom'
        for x, y in path[::2]:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(x)
            ps.pose.position.y = float(y)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.path_pub.publish(msg)

    def _publish_occupancy(self):
        grid = OccupancyGrid()
        grid.header.stamp = self._now().to_msg()
        grid.header.frame_id = 'odom'
        res = float(self.xs[1] - self.xs[0]) if self.xs.size > 1 else GRID_RES
        grid.info.resolution = res
        grid.info.width = int(self.xs.size)
        grid.info.height = int(self.ys.size)
        grid.info.origin.position.x = float(self.xs[0] - 0.5 * res)
        grid.info.origin.position.y = float(self.ys[0] - 0.5 * res)
        grid.info.origin.orientation.w = 1.0
        data = np.where(self.occ > 0, 100, 0).astype(np.int8).ravel(order='C')
        grid.data = data.tolist()
        self.map_pub.publish(grid)

    def _publish_lookahead(self, xy):
        ps = PoseStamped()
        ps.header.stamp = self._now().to_msg()
        ps.header.frame_id = 'odom'
        ps.pose.position.x = float(xy[0])
        ps.pose.position.y = float(xy[1])
        ps.pose.position.z = 1.0
        ps.pose.orientation.w = 1.0
        self.look_pub.publish(ps)
        mk = Marker()
        mk.header = ps.header
        mk.ns = 'lookahead'
        mk.id = 0
        mk.type = Marker.SPHERE
        mk.action = Marker.ADD
        mk.pose = ps.pose
        mk.scale.x = mk.scale.y = mk.scale.z = 0.45
        mk.color.r = 1.0
        mk.color.g = 0.55
        mk.color.b = 0.0
        mk.color.a = 0.95
        self.marker_pub.publish(mk)

    def _tick(self):
        if not self._have_odom:
            return
        path = self._plan()
        if not path:
            if not self._blocked:
                self.interventions += 1
                self._blocked = True
            look = self.global_goal
        else:
            self._blocked = False
            self.path = path
            look = lookahead_on_path(
                path,
                (self.x, self.y),
                float(self.get_parameter('lookahead_m').value),
            )
            if look is None:
                look = self.global_goal
        self._publish_lookahead(look)
        if path:
            self._publish_path(path)
        if self._have_scan:
            self._publish_occupancy()
        self.count_pub.publish(Int32(data=self.interventions))


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
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
