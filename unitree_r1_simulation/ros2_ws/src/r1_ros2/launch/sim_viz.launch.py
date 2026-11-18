import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("r1_description")
    with open(os.path.join(share, "urdf", "r1_brainco.urdf"), "r", encoding="utf-8") as f:
        robot_description = f.read()

    return LaunchDescription([
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("camera", default_value="true", description="bridge the head camera stream"),
        DeclareLaunchArgument("camera_host", default_value="127.0.0.1"),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            output="screen",
        ),
        Node(package="r1_ros2", executable="state_bridge", output="screen"),
        Node(
            package="r1_ros2",
            executable="camera_bridge",
            parameters=[{"host": LaunchConfiguration("camera_host")}],
            condition=IfCondition(LaunchConfiguration("camera")),
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", os.path.join(share, "rviz", "r1.rviz")],
            condition=IfCondition(LaunchConfiguration("rviz")),
            output="screen",
        ),
    ])
