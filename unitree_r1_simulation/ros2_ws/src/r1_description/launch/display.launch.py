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

    gui = LaunchConfiguration("gui")
    rviz = LaunchConfiguration("rviz")
    return LaunchDescription([
        DeclareLaunchArgument("gui", default_value="true", description="start joint_state_publisher_gui"),
        DeclareLaunchArgument("rviz", default_value="true", description="start RViz"),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": robot_description}],
            output="screen",
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            condition=IfCondition(gui),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", os.path.join(share, "rviz", "r1.rviz")],
            condition=IfCondition(rviz),
            output="screen",
        ),
    ])
