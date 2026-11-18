# syntax=docker/dockerfile:1

FROM ros:humble-ros-base-jammy

ARG UNITREE_ROS2_COMMIT=668d1ec5a05d1c38d3306bdca7d59f2ba3581a88
ARG UNITREE_ROS_COMMIT=5994d4faef0a9cadd3287f8de0199a67eeb2a259

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-rmw-cyclonedds-cpp \
        ros-humble-rosidl-generator-dds-idl \
        libyaml-cpp-dev \
        ros-humble-robot-state-publisher \
        ros-humble-joint-state-publisher-gui \
        ros-humble-rviz2 \
        ros-humble-xacro \
        python3-pip python3-numpy python3-opencv python3-zmq \
        iproute2 git \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install "trimesh>=4.4,<5"

RUN git clone https://github.com/unitreerobotics/unitree_ros2.git /tmp/unitree_ros2 \
    && git -C /tmp/unitree_ros2 checkout ${UNITREE_ROS2_COMMIT} \
    && mkdir -p /opt/unitree_ws/src \
    && cp -r /tmp/unitree_ros2/cyclonedds_ws/src/unitree /opt/unitree_ws/src/ \
    && rm -rf /tmp/unitree_ros2 \
    && source /opt/ros/humble/setup.bash \
    && cd /opt/unitree_ws \
    && colcon build --event-handlers console_cohesion+ \
    && rm -rf build log

COPY ros2_ws/src /opt/r1_ws/src
COPY tools/build_r1_brainco_model.py /opt/r1_tools/build_r1_brainco_model.py
COPY r1_sim/r1_sim/joint_map.py /opt/r1_tools/sim_joint_map.py
RUN git clone --filter=blob:none --no-checkout https://github.com/unitreerobotics/unitree_ros.git /tmp/unitree_ros \
    && cd /tmp/unitree_ros \
    && git sparse-checkout init --cone \
    && git sparse-checkout set robots/r1_description robots/g1_with_brainco_hand \
    && git checkout ${UNITREE_ROS_COMMIT} \
    && python3 /opt/r1_tools/build_r1_brainco_model.py \
        --r1-dir robots/r1_description \
        --brainco-urdf robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf \
        --joint-map /opt/r1_tools/sim_joint_map.py \
        --out-dir /tmp/r1_model \
        --mesh-prefix package://r1_description/meshes/ \
    && mkdir -p /opt/r1_ws/src/r1_description/urdf \
    && cp /tmp/r1_model/r1_brainco.urdf /opt/r1_ws/src/r1_description/urdf/ \
    && cp -r /tmp/r1_model/meshes /opt/r1_ws/src/r1_description/ \
    && rm -rf /tmp/unitree_ros /tmp/r1_model
RUN cd /opt/r1_tools && python3 -c "\
import importlib.util, sys; sys.path.insert(0, '/opt/r1_ws/src/r1_ros2'); \
spec = importlib.util.spec_from_file_location('sim_joint_map', '/opt/r1_tools/sim_joint_map.py'); \
sim = importlib.util.module_from_spec(spec); spec.loader.exec_module(sim); \
import r1_ros2.joints as ros; \
assert sim.R1_BODY_JOINT_SLOTS == ros.BODY_JOINT_SLOTS, 'body joint map differs'; \
assert tuple(sim.HAND_MOTOR_JOINTS) == tuple(ros.HAND_MOTOR_JOINTS), 'hand motor order differs'; \
assert tuple(sim.HAND_MOTOR_RANGES) == tuple(ros.HAND_MOTOR_RANGES), 'hand motor ranges differ'; \
assert dict(sim.HAND_COUPLED_JOINTS) == dict(ros.HAND_COUPLED_JOINTS), 'hand coupling differs'; \
assert [sim.hand_joint_names(s) for s in sim.HAND_SIDES] == [ros.hand_joint_names(s) for s in ros.HAND_SIDES]; \
print('joint maps identical')"
RUN source /opt/ros/humble/setup.bash \
    && source /opt/unitree_ws/install/setup.bash \
    && cd /opt/r1_ws \
    && colcon build --event-handlers console_cohesion+ \
    && rm -rf build log \
    && source /opt/r1_ws/install/setup.bash \
    && check_urdf_out=$(python3 -c "import xml.etree.ElementTree as ET; r = ET.parse('/opt/r1_ws/install/r1_description/share/r1_description/urdf/r1_brainco.urdf').getroot(); print(len(r.findall('joint')))") \
    && echo "r1_description installed (${check_urdf_out} joints)"

COPY docker/ros2/r1_ros_env.sh /opt/r1_ros_env.sh
COPY docker/ros2/ros_entrypoint.sh /ros_entrypoint_r1.sh
RUN chmod +x /ros_entrypoint_r1.sh \
    && echo "source /opt/r1_ros_env.sh" >> /root/.bashrc

ENV DDS_DOMAIN_ID=1 \
    DDS_INTERFACE=lo \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ENTRYPOINT ["/ros_entrypoint_r1.sh"]
CMD ["ros2", "launch", "r1_ros2", "sim_viz.launch.py"]
