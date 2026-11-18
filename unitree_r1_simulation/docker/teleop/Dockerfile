# syntax=docker/dockerfile:1

FROM ubuntu:22.04

ARG MINIFORGE_VERSION=26.7.2-0
ARG XR_TELEOPERATE_COMMIT=817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b
ARG UNITREE_SDK2_PYTHON_COMMIT=814556d15970dd2ecf1c9984e845ca02ab07e206

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_CONSTRAINT=/opt/teleop/constraints.txt \
    PATH=/opt/conda/envs/tv/bin:/opt/conda/bin:${PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git openssl iproute2 procps \
        libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda create -y -n tv -c conda-forge python=3.10 pinocchio=3.1.0 numpy=1.26.4 \
    && /opt/conda/bin/conda clean -afy

COPY docker/teleop/constraints.txt /opt/teleop/constraints.txt

RUN git clone https://github.com/unitreerobotics/xr_teleoperate.git /opt/xr_teleoperate \
    && cd /opt/xr_teleoperate \
    && git checkout ${XR_TELEOPERATE_COMMIT} \
    && git submodule update --init --depth 1

RUN pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-deps -e /opt/xr_teleoperate/teleop/teleimager \
    && pip install logging_mp opencv-python pyyaml pyzmq
RUN pip install -e /opt/xr_teleoperate/teleop/televuer
RUN pip install --no-deps -e /opt/xr_teleoperate/teleop/robot_control/dex-retargeting \
    && pip install "pytransform3d>=3.5.0" nlopt "trimesh>=4.4.0" "anytree>=2.12.0" "pyyaml>=6.0.0" "lxml>=5.2.2" six
RUN pip install -r /opt/xr_teleoperate/requirements.txt
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git /opt/unitree_sdk2_python \
    && git -C /opt/unitree_sdk2_python checkout ${UNITREE_SDK2_PYTHON_COMMIT} \
    && pip install -e /opt/unitree_sdk2_python

ARG UNITREE_ROS_COMMIT=5994d4faef0a9cadd3287f8de0199a67eeb2a259
RUN pip install fast-simplification==0.1.11
COPY tools/build_r1_brainco_model.py tools/build_vr_meshes.py /opt/r1_tools/
COPY r1_sim/r1_sim/joint_map.py /opt/r1_tools/joint_map.py
RUN git clone --filter=blob:none --no-checkout https://github.com/unitreerobotics/unitree_ros.git /tmp/unitree_ros \
    && cd /tmp/unitree_ros \
    && git sparse-checkout init --cone \
    && git sparse-checkout set robots/r1_description robots/g1_with_brainco_hand \
    && git checkout ${UNITREE_ROS_COMMIT} \
    && python /opt/r1_tools/build_r1_brainco_model.py \
        --r1-dir robots/r1_description \
        --brainco-urdf robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf \
        --joint-map /opt/r1_tools/joint_map.py \
        --out-dir /opt/r1_model \
        --strip-mimic \
    && rm -rf /tmp/unitree_ros \
    && python /opt/r1_tools/build_vr_meshes.py --urdf /opt/r1_model/r1_brainco.urdf --out /opt/r1_vr/robot_meshes.npz

COPY docker/teleop/start_teleop.sh /usr/local/bin/start_teleop.sh
COPY docker/teleop/wait_for_sim.py docker/teleop/smoke_test.py /opt/teleop/
COPY ros2_ws/src/r1_ros2/r1_ros2/crc.py /opt/teleop/r1_crc.py
COPY config/teleop/cam_config_client.yaml /opt/xr_teleoperate/teleop/teleimager/cam_config_client.yaml
COPY r1_sim /opt/r1_sim
COPY docker/teleop/patches /opt/teleop/patches
RUN cd /opt/xr_teleoperate && git apply -v /opt/teleop/patches/r1_teleop.patch \
    && git -C teleop/televuer apply -v /opt/teleop/patches/televuer_r1.patch

WORKDIR /opt/xr_teleoperate/teleop
RUN chmod +x /usr/local/bin/start_teleop.sh \
    && (pip check || true) \
    && python /opt/teleop/smoke_test.py \
    && chmod -R a+rwX /opt/xr_teleoperate/teleop

ENV DDS_DOMAIN_ID=1 \
    DDS_INTERFACE=lo \
    XR_TELEOP_CERT=/certs/cert.pem \
    XR_TELEOP_KEY=/certs/key.pem \
    HOME=/tmp/teleop-home \
    MPLCONFIGDIR=/tmp/teleop-home/.config/matplotlib

ENTRYPOINT ["/usr/local/bin/start_teleop.sh"]
