import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    launch_dir = os.path.join(get_package_share_directory('rotor_fail'), 'launch')

    def include(name, args=None):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, name)),
            launch_arguments=(args or {}).items())

    sim = include('launch_sim.launch.py', {
        'world_name': LaunchConfiguration('world_name'),
        'drone_model': LaunchConfiguration('drone_model'),
        'headless': LaunchConfiguration('headless'),
        'rtf': LaunchConfiguration('rtf'),
        'px4_dir': LaunchConfiguration('px4_dir'),
    })

    def start_qgc(context):
        path = os.path.expanduser(context.perform_substitution(LaunchConfiguration('qgc_path')))
        if context.perform_substitution(LaunchConfiguration('headless')) == 'true' or not os.path.isfile(path):
            return []  # no QGC: headless, or no AppImage at qgc_path
        return [ExecuteProcess(cmd=[path], additional_env={'QT_QPA_PLATFORM': 'xcb'}, output='screen')]

    pftc = Node(package='rotor_fail', executable='pftc_node.py', output='screen',
                parameters=[{'use_sim_time': True, 'state': LaunchConfiguration('state'),
                             'debug_csv': LaunchConfiguration('debug_csv')}],
                condition=IfCondition(LaunchConfiguration('ftc')))

    return LaunchDescription([
        DeclareLaunchArgument('world_name', default_value='empty'),
        DeclareLaunchArgument('drone_model', default_value='custom_x500'),
        DeclareLaunchArgument('headless', default_value='false', description='true = no Gazebo GUI, no QGC'),
        DeclareLaunchArgument('rtf', default_value='1.0', description='real-time factor of the sim (> 1 = faster, benchmarks)'),
        DeclareLaunchArgument('px4_dir', default_value='~/PX4-Autopilot', description='PX4-Autopilot source directory (built with make px4_sitl)'),
        DeclareLaunchArgument('qgc_path', default_value='~/QGroundControl.AppImage', description='QGroundControl AppImage; skipped if the file does not exist'),
        DeclareLaunchArgument('inject', default_value='true', description='true = start the random fault injector'),
        DeclareLaunchArgument('state', default_value='gyro',
                              description="policy state source: gyro (default, onboard gyro attitude + GPS/accel KF), ekf, gt (diagnosis)"),
        DeclareLaunchArgument('debug_csv', default_value='', description='sim only: CSV of the onboard estimator error vs ground truth'),
        DeclareLaunchArgument('ftc', default_value='true',
                              description='true = hand the motors to the learned policy (policies/x500v2_i025a) when a failure is detected'),
        sim,
        include('bridge.launch.py'),
        include('failure.launch.py', {'inject': LaunchConfiguration('inject')}),
        pftc,
        OpaqueFunction(function=start_qgc),
    ])
