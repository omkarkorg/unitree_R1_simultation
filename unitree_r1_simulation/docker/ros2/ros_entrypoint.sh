#!/usr/bin/env bash
set -e
source /opt/r1_ros_env.sh
echo "[r1-ros2] ROS_DOMAIN_ID=${ROS_DOMAIN_ID} RMW=${RMW_IMPLEMENTATION} DDS interface=${DDS_INTERFACE:-lo}"
exec "$@"
