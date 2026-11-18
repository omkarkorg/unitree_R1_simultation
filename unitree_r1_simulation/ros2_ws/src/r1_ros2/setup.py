from glob import glob

from setuptools import setup

package_name = "r1_ros2"

setup(
    name=package_name,
    version="1.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="r1 workspace",
    maintainer_email="user@example.com",
    description="ROS 2 Humble bridges and examples for the Unitree R1 + BrainCo hands",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "state_bridge = r1_ros2.state_bridge:main",
            "camera_bridge = r1_ros2.camera_bridge:main",
            "arm_demo = r1_ros2.arm_demo:main",
        ],
    },
)
