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


def wait_for_create_service(world: str, timeout_sec: float = 90.0) -> None:
    needle = f'/world/{world}/create'
    deadline = time.time() + timeout_sec
    last = ''
    while time.time() < deadline:
        proc = run(['gz', 'service', '-l'], check=False)
        last = proc.stdout or ''
        if needle in last:
            # UserCommands + GPU lidar come up after the world name exists.
            time.sleep(2.5)
            return
        time.sleep(0.4)
    raise RuntimeError(
        f'timed out waiting for {needle}. gz service -l tail:\n{last[-2000:]}'
    )


def try_remove(world: str, name: str) -> None:
    run(
        [
            'gz', 'service', '-s', f'/world/{world}/remove',
            '--reqtype', 'gz.msgs.Entity',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '4000',
            '--req', f'name: "{name}", type: 6',
        ],
        check=False,
    )


def try_create(world: str, sdf: Path, name: str, x: float, y: float, z: float):
    return run(
        [
            'gz', 'service', '-s', f'/world/{world}/create',
            '--reqtype', 'gz.msgs.EntityFactory',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '20000',
            '--req',
            f'sdf_filename: "{sdf}", name: "{name}", '
            f'pose: {{position: {{x: {x}, y: {y}, z: {z}}}}}',
        ],
        check=False,
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
    wait_for_create_service(args.world)
    run(['xacro', str(xacro), '-o', str(urdf)])
    converted = run(['gz', 'sdf', '-p', str(urdf)])
    if not converted.stdout.strip():
        print(converted.stderr, file=sys.stderr)
        raise RuntimeError('gz sdf -p produced empty drone SDF')
    sdf.write_text(converted.stdout)

    last = None
    poses = [
        (args.x, args.y, args.z),
        (args.x, args.y, args.z + 0.4),
        (args.x, args.y + 0.4, args.z),
        (args.x + 0.4, args.y, args.z),
    ]
    for attempt, (x, y, z) in enumerate(poses * 3):
        try_remove(args.world, 'drone_robot')
        time.sleep(0.3)
        last = try_create(args.world, sdf, 'drone_robot', x, y, z)
        out = (last.stdout or '') + (last.stderr or '')
        if 'data: true' in out:
            time.sleep(0.4)
            run(
                [
                    'gz', 'topic', '-t', '/drone_robot/enable',
                    '-m', 'gz.msgs.Boolean', '-p', 'data: true',
                ],
                check=False,
            )
            print(
                f'spawned and enabled drone_robot in {args.world} '
                f'at ({x:.2f}, {y:.2f}, {z:.2f}) (attempt {attempt + 1})'
            )
            return
        print(f'spawn attempt {attempt + 1} failed: {out.strip()[:400]}', file=sys.stderr)
        time.sleep(0.8)

    print(last.stdout if last else '', last.stderr if last else '', file=sys.stderr)
    print('worlds:\n' + _list_worlds(), file=sys.stderr)
    raise RuntimeError(
        f'gz create did not return data: true for /world/{args.world}/create'
    )


if __name__ == '__main__':
    main()
