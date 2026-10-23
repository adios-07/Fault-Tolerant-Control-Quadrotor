from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('inject', default_value='true',
                              description='true = start the random fault injector (off for benchmark runs)'),
        Node(
            package='rotor_fail',
            executable='failure_detection.py',
            output='screen',
            parameters=[{'use_sim_time': True}],
        ),
        Node(
            package='rotor_fail',
            executable='fault_injector.py',
            output='screen',
            parameters=[{'use_sim_time': True}],
            condition=IfCondition(LaunchConfiguration('inject')),
        )
    ])
