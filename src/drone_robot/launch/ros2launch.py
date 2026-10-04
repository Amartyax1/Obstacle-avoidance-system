"""
Gazebo, Harmonic gz.transport bridge, spawn, ONNX inference, live A* planner.

  ros2 launch drone_robot ros2launch.py
  ros2 launch drone_robot ros2launch.py world:=warehouse spawn_x:=-5.0
  ros2 launch drone_robot ros2launch.py world:=eval_1 spawn_x:=-5.0
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory('drone_robot')
    world = context.perform_substitution(LaunchConfiguration('world'))
    spawn_x = context.perform_substitution(LaunchConfiguration('spawn_x'))
    spawn_y = context.perform_substitution(LaunchConfiguration('spawn_y'))
    spawn_z = context.perform_substitution(LaunchConfiguration('spawn_z'))
    loop_waypoints = context.perform_substitution(
        LaunchConfiguration('loop_waypoints')
    ) == 'true'
    use_planner = context.perform_substitution(
        LaunchConfiguration('use_planner')
    ) == 'true'

    world_file = os.path.join(pkg_share, 'worlds', f'{world}.sdf')
    xacro_file = os.path.join(pkg_share, 'model', 'robot.xacro')
    params_file = os.path.join(pkg_share, 'config', 'params.yaml')
    rviz_file = os.path.join(pkg_share, 'config', 'planner.rviz')
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]), value_type=str
    )

    infer_extra = {
        'use_sim_time': True,
        'loop_waypoints': loop_waypoints,
        'use_planner': use_planner,
    }
    planner_params = {
        'use_sim_time': True,
        'online_map': True,
        'lookahead_m': 2.0,
    }
    if world == 'explore':
        infer_extra.update({
            'target_x': 40.0,
            'target_y': 0.0,
            'waypoint_xs': [40.0],
            'waypoint_ys': [0.0],
            'loop_waypoints': False,
        })
        planner_params.update({
            'xmin': -52.0,
            'xmax': 52.0,
            'ymin': -12.0,
            'ymax': 12.0,
            'goal_x': 40.0,
            'goal_y': 0.0,
        })
    else:
        grid_name = 'warehouse_grid.npz' if world == 'warehouse' else f'{world}_grid.npz'
        planner_params.update({
            'xmin': -12.0,
            'xmax': 32.0,
            'ymin': -14.0,
            'ymax': 14.0,
            'goal_x': 26.0,
            'goal_y': 0.0,
            'grid_path': os.path.join(pkg_share, 'config', grid_name),
        })

    spawn = Node(
        package='drone_robot',
        executable='spawn_drone.py',
        name='spawn_drone',
        output='screen',
        arguments=['--world', world, '--x', spawn_x, '--y', spawn_y, '--z', spawn_z],
    )
    inference = Node(
        package='drone_robot',
        executable='inference_node.py',
        name='drone_inference_node',
        parameters=[params_file, infer_extra],
        output='screen',
    )
    planner = Node(
        package='drone_robot',
        executable='planner_node.py',
        name='planner_node',
        parameters=[planner_params],
        output='screen',
        condition=IfCondition(LaunchConfiguration('use_planner')),
    )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_file],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
        output='screen',
    )
    gz_gui = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-v', '2', world_file],
        output='screen',
        condition=UnlessCondition(LaunchConfiguration('headless')),
    )
    gz_headless = ExecuteProcess(
        cmd=['gz', 'sim', '-s', '-r', '-v', '2', world_file],
        output='screen',
        condition=IfCondition(LaunchConfiguration('headless')),
    )
    after_spawn = [inference]
    if use_planner:
        after_spawn.append(planner)
    return [
        gz_gui,
        gz_headless,
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            parameters=[{'robot_description': robot_description,
                         'use_sim_time': True}],
            output='screen',
        ),
        Node(
            package='drone_robot',
            executable='gz_bridge_node.py',
            name='harmonic_gz_bridge',
            output='screen',
        ),
        TimerAction(period=3.0, actions=[spawn]),
        rviz,
        RegisterEventHandler(
            OnProcessExit(target_action=spawn, on_exit=after_spawn),
        ),
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory('drone_robot')
    models = os.path.join(pkg_share, 'models')
    factory_models = str(Path.home() / '.gazebo' / 'models' / 'factory' / 'models')
    existing = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    resource_path = models + ':' + factory_models
    if existing:
        resource_path = resource_path + ':' + existing

    return LaunchDescription([
        DeclareLaunchArgument(
            'world', default_value='explore',
            description='World stem under worlds/ (explore, warehouse, eval_1..5, forest, factory).'
        ),
        DeclareLaunchArgument(
            'headless', default_value='false',
            description='Run Gazebo server-only (no GUI).'
        ),
        DeclareLaunchArgument(
            'spawn_x', default_value='-36.0',
            description='Spawn X (explore west; warehouse west is -5).'
        ),
        DeclareLaunchArgument('spawn_y', default_value='0.0'),
        DeclareLaunchArgument('spawn_z', default_value='1.0'),
        DeclareLaunchArgument(
            'loop_waypoints', default_value='false',
            description='After the last goal, cycle back to the first.'
        ),
        DeclareLaunchArgument(
            'use_planner', default_value='true',
            description='Live occupancy A* lookahead (never /cmd_vel).'
        ),
        DeclareLaunchArgument(
            'use_rviz', default_value='false',
            description='Show global path and lookahead marker.'
        ),
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', factory_models),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_path),
        OpaqueFunction(function=_setup),
    ])
