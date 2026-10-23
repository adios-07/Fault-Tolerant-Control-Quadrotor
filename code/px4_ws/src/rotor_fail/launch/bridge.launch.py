import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

# ROS <-> PX4 / Gazebo plumbing:
#   - MicroXRCEAgent : uXRCE-DDS bridge for PX4 <-> ROS 2 (/fmu/* topics)
#   - ros_gz_bridge  : Gazebo <-> ROS topics listed in config/gz_bridge.yaml


def generate_launch_description():
    bridge_params = os.path.join(get_package_share_directory('rotor_fail'), 'config', 'gz_bridge.yaml')

    micro_dds_agent = ExecuteProcess(
        cmd=['MicroXRCEAgent', 'udp4', '-p', '8888'],
        output='screen'
    )

    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['--ros-args', '-p', f'config_file:={bridge_params}'],
        additional_env={'GZ_IP': '127.0.0.1'},  # must match the sim's GZ_IP or no gz topics are seen
        output='screen'
    )

    return LaunchDescription([micro_dds_agent, bridge_node])
