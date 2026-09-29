#!/usr/bin/env python3
"""Wait for the gz world, convert the xacro to SDF, then spawn and enable."""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from ament_index_python.packages import get_package_share_directory


def run(cmd, check=True):
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def _list_worlds() -> str:
    proc = subprocess.run(
        [
            'gz', 'service', '-s', '/gazebo/worlds',
            '--reqtype', 'gz.msgs.Empty',
            '--reptype', 'gz.msgs.StringMsg_V',
            '--timeout', '2000',
            '--req', '',
        ],
        capture_output=True,
        text=True,
    )
    return (proc.stdout or '') + (proc.stderr or '')


def wait_for_world(world: str, timeout_sec: float = 90.0) -> None:
    deadline = time.time() + timeout_sec
    last = ''
    while time.time() < deadline:
        last = _list_worlds()
        if world in last:
            return
        time.sleep(0.5)
    raise RuntimeError(
        f'timed out waiting for gz world {world}. /gazebo/worlds said:\n{last}'
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', default='factory')
    parser.add_argument('--x', type=float, default=-5.0)
    parser.add_argument('--y', type=float, default=0.0)
    parser.add_argument('--z', type=float, default=1.0)
    args, _unknown = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    pkg = Path(get_package_share_directory('drone_robot'))
    xacro = pkg / 'model' / 'robot.xacro'
    urdf = Path('/tmp/drone_robot.urdf')
    sdf = Path('/tmp/drone_robot.sdf')

    wait_for_world(args.world)
    run(['xacro', str(xacro), '-o', str(urdf)])
    sdf.write_text(run(['gz', 'sdf', '-p', str(urdf)]).stdout)

    create = run([
        'gz', 'service', '-s', f'/world/{args.world}/create',
        '--reqtype', 'gz.msgs.EntityFactory',
        '--reptype', 'gz.msgs.Boolean',
        '--timeout', '8000',
        '--req',
        f'sdf_filename: "{sdf}", name: "drone_robot", '
        f'pose: {{position: {{x: {args.x}, y: {args.y}, z: {args.z}}}}}',
    ])
    if 'data: true' not in (create.stdout or ''):
        print(create.stdout, create.stderr, file=sys.stderr)
        print('worlds:\n' + _list_worlds(), file=sys.stderr)
        raise RuntimeError(
            f'gz create did not return data: true for /world/{args.world}/create'
        )

    time.sleep(0.5)
    run([
        'gz', 'topic', '-t', '/drone_robot/enable',
        '-m', 'gz.msgs.Boolean', '-p', 'data: true',
    ], check=False)
    print(
        f'spawned and enabled drone_robot in {args.world} '
        f'at ({args.x:.2f}, {args.y:.2f}, {args.z:.2f})'
    )


if __name__ == '__main__':
    main()
