"""
Gazebo, Harmonic gz.transport bridge, spawn, and ONNX inference.

Usage:
  ros2 launch drone_robot ros2launch.py
  ros2 launch drone_robot ros2launch.py world:=forest spawn_x:=0.0
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

    world_file = os.path.join(pkg_share, 'worlds', f'{world}.sdf')
    xacro_file = os.path.join(pkg_share, 'model', 'robot.xacro')
    params_file = os.path.join(pkg_share, 'config', 'params.yaml')
    robot_description = ParameterValue(
        Command(['xacro ', xacro_file]), value_type=str
    )

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
        parameters=[params_file, {
            'use_sim_time': True,
            'loop_waypoints': loop_waypoints,
        }],
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
        spawn,
        RegisterEventHandler(
            OnProcessExit(target_action=spawn, on_exit=[inference]),
        ),
    ]


def generate_launch_description():
    factory_models = str(Path.home() / '.gazebo' / 'models' / 'factory' / 'models')
    resource_path = factory_models
    existing = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    if existing:
        resource_path = factory_models + ':' + existing

    return LaunchDescription([
        DeclareLaunchArgument(
            'world', default_value='factory',
            description='World stem under worlds/ (factory or forest).'
        ),
        DeclareLaunchArgument(
            'headless', default_value='false',
            description='Run Gazebo server-only (no GUI).'
        ),
        DeclareLaunchArgument(
            'spawn_x', default_value='-5.0',
            description='Spawn X at the west end of the factory.'
        ),
        DeclareLaunchArgument('spawn_y', default_value='0.0'),
        DeclareLaunchArgument('spawn_z', default_value='1.0'),
        DeclareLaunchArgument(
            'loop_waypoints', default_value='true',
            description='After the last factory goal, cycle back to the first.'
        ),
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', factory_models),
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_path),
        OpaqueFunction(function=_setup),
    ])
