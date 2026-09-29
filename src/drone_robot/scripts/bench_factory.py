#!/usr/bin/env python3
"""90 s warehouse bench: west -> east -> west.

Assumes `ros2 launch drone_robot ros2launch.py` is already running.

    python3 src/drone_robot/scripts/bench_factory.py
    python3 src/drone_robot/scripts/bench_factory.py --grid .../eval_1_grid.npz --world eval_1
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Int32

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
        self.interventions = 0
        self.create_subscription(Odometry, '/odom', self._on_odom, qos_profile_sensor_data)
        self.create_subscription(Twist, '/cmd_vel', self._on_cmd, 10)
        self.create_subscription(Int32, '/planner/interventions', self._on_n, 10)

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        self.samples.append((time.time(), p.x, p.y, p.z))

    def _on_cmd(self, msg: Twist):
        self.cmds.append((time.time(), msg.linear.x))

    def _on_n(self, msg: Int32):
        self.interventions = int(msg.data)


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
    print(f"astar_interventions={r.get('interventions', 0)}")
    ok = (
        r['east_t'] is not None
        and r['west_t'] is not None
        and r['zmax'] < 1.8
        and not r['crash']
    )
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


def _load_goals(grid_path: str | None):
    east, west = EAST, WEST
    if grid_path:
        data = np.load(grid_path, allow_pickle=True)
        g = np.asarray(data['goal'], dtype=float)
        s = np.asarray(data['spawn'], dtype=float)
        east = (float(g[0]), float(g[1]))
        west = (float(s[0]), float(s[1]))
    return east, west


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', default='')
    parser.add_argument('--world', default='warehouse')
    parser.add_argument('--csv', default='')
    parser.add_argument('--timeout', type=float, default=TIMEOUT_SEC)
    args = parser.parse_args()
    global EAST, WEST
    EAST, WEST = _load_goals(args.grid or None)

    rclpy.init()
    node = FactoryBench()
    deadline = time.time() + float(args.timeout)
    print(f'waiting up to {args.timeout:.0f}s for /odom ...', flush=True)
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
    samples, cmds, n_int = node.samples, node.cmds, node.interventions
    node.destroy_node()
    rclpy.shutdown()
    if not samples:
        print('FAIL: no /odom')
        return 1
    r = summarize(samples, cmds)
    r['interventions'] = n_int
    if args.csv:
        path = Path(args.csv)
        new = not path.is_file()
        with path.open('a', newline='') as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    'world', 'success', 'crash', 'astar_interventions',
                    'time_sec', 'east_t', 'west_t', 'zmin', 'zmax',
                ],
            )
            if new:
                w.writeheader()
            ok = r['east_t'] is not None and r['west_t'] is not None and not r['crash']
            w.writerow({
                'world': args.world,
                'success': int(ok),
                'crash': int(r['crash']),
                'astar_interventions': n_int,
                'time_sec': f"{r['dur']:.2f}",
                'east_t': '' if r['east_t'] is None else f"{r['east_t']:.2f}",
                'west_t': '' if r['west_t'] is None else f"{r['west_t']:.2f}",
                'zmin': f"{r['zmin']:.2f}",
                'zmax': f"{r['zmax']:.2f}",
            })
    return print_report(r)


if __name__ == '__main__':
    sys.exit(main())
