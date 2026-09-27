#!/usr/bin/env python3
"""90 s Gazebo factory bench: west (-5,0) -> east (26,0) -> west.

Assumes `ros2 launch drone_robot ros2launch.py` is already running.

    python3 src/drone_robot/scripts/bench_factory.py
"""

from __future__ import annotations

import math
import sys
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

EAST = (26.0, 0.0)
WEST = (-5.0, 0.0)
TOL = 0.8
TIMEOUT_SEC = 90.0
FREEZE_VX = 0.05
FREEZE_SEC = 5.0
CRASH_Z = 0.2
ODOM_JUMP = 20.0


class FactoryBench(Node):
    def __init__(self):
        super().__init__('factory_bench')
        self.samples: list[tuple[float, float, float, float]] = []
        self.cmds: list[tuple[float, float]] = []
        self.create_subscription(Odometry, '/odom', self._on_odom, qos_profile_sensor_data)
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd, 10)

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self.samples.append((time.time(), p.x, p.y, p.z))

    def _on_cmd(self, msg: Twist):
        self.cmds.append((time.time(), msg.linear.x))


def _first_reach(samples, goal, t0):
    gx, gy = goal
    closest = math.inf
    t_hit = None
    for t, x, y, _z in samples:
        d = math.hypot(x - gx, y - gy)
        if d < closest:
            closest = d
        if t_hit is None and d < TOL:
            t_hit = t - t0
    return t_hit, closest


def _freeze_sec(cmds, t0, t1):
    if not cmds:
        return 0.0
    longest = 0.0
    run = 0.0
    prev = None
    for t, vx in cmds:
        if t < t0 or t > t1:
            continue
        if prev is not None and abs(vx) < FREEZE_VX:
            run += t - prev
            longest = max(longest, run)
        else:
            run = 0.0
        prev = t
    return longest


def summarize(samples, cmds) -> dict:
    t0 = samples[0][0]
    xs = [s[1] for s in samples]
    ys = [s[2] for s in samples]
    zs = [s[3] for s in samples]
    dur = samples[-1][0] - t0
    east_t, east_c = _first_reach(samples, EAST, t0)
    west_samples = samples
    west_t = None
    west_c = math.inf
    if east_t is not None:
        west_samples = [s for s in samples if s[0] >= t0 + east_t]
        west_t, west_c = _first_reach(west_samples, WEST, t0)
    freeze = _freeze_sec(cmds, t0, samples[-1][0])
    jumps = 0
    for i in range(1, len(samples)):
        dx = samples[i][1] - samples[i - 1][1]
        dy = samples[i][2] - samples[i - 1][2]
        if math.hypot(dx, dy) > ODOM_JUMP:
            jumps += 1
    crash = min(zs) < CRASH_Z or jumps > 0
    return {
        'dur': dur,
        'n': len(samples),
        'end': (xs[-1], ys[-1], zs[-1]),
        'zmin': min(zs),
        'zmax': max(zs),
        'east_t': east_t,
        'east_c': east_c,
        'west_t': west_t,
        'west_c': west_c,
        'freeze': freeze,
        'crash': crash,
        'jumps': jumps,
        'xmax': max(xs),
        'xmin': min(xs),
    }


def print_report(r: dict) -> int:
    def fmt_t(v):
        return f'{v:.1f}s' if v is not None else 'MISS'

    print('--- factory bench ---')
    print(f"samples={r['n']} dur={r['dur']:.1f}s  end=({r['end'][0]:.2f}, {r['end'][1]:.2f}, {r['end'][2]:.2f})")
    print(f"east (26,0): {fmt_t(r['east_t'])}  closest={r['east_c']:.2f} m")
    print(f"west (-5,0) after east: {fmt_t(r['west_t'])}  closest={r['west_c']:.2f} m")
    print(f"z min={r['zmin']:.2f} max={r['zmax']:.2f}")
    print(f"x range=[{r['xmin']:.2f}, {r['xmax']:.2f}]")
    print(f"freeze {r['freeze']:.1f}s  crash={r['crash']} odom_jumps={r['jumps']}")
    ok = (
        r['east_t'] is not None
        and r['west_t'] is not None
        and r['zmax'] < 1.8
        and not r['crash']
    )
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


def main():
    rclpy.init()
    node = FactoryBench()
    deadline = time.time() + TIMEOUT_SEC
    print(f'waiting up to {TIMEOUT_SEC:.0f}s for /odom ...', flush=True)
    east_done_at = None
    west_done = False
    while time.time() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if not node.samples:
            continue
        r = summarize(node.samples, node.cmds)
        if r['east_t'] is not None and east_done_at is None:
            east_done_at = time.time()
            print(f"east reached in {r['east_t']:.1f}s", flush=True)
        if r['west_t'] is not None and not west_done:
            west_done = True
            print(f"west return in {r['west_t']:.1f}s", flush=True)
            time.sleep(1.0)
            break
    node.destroy_node()
    rclpy.shutdown()
    if not node.samples:
        print('FAIL: no /odom')
        return 1
    return print_report(summarize(node.samples, node.cmds))


if __name__ == '__main__':
    sys.exit(main())
