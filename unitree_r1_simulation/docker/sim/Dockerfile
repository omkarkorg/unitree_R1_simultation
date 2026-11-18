# syntax=docker/dockerfile:1

FROM isaaclab_base

ARG CYCLONEDDS_VERSION=0.10.5
ARG UNITREE_SDK2_PYTHON_COMMIT=814556d15970dd2ecf1c9984e845ca02ab07e206
ARG UNITREE_ROS_COMMIT=5994d4faef0a9cadd3287f8de0199a67eeb2a259

USER root
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV CYCLONEDDS_HOME=/opt/cyclonedds

RUN apt-get update && apt-get install -y --no-install-recommends iproute2 iputils-ping \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch ${CYCLONEDDS_VERSION} https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds \
    && cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build \
        -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=${CYCLONEDDS_HOME} \
        -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DBUILD_IDLC_TESTING=OFF -DENABLE_SHM=OFF \
    && cmake --build /tmp/cyclonedds/build --target install -j"$(nproc)" \
    && echo "${CYCLONEDDS_HOME}/lib" > /etc/ld.so.conf.d/cyclonedds.conf && ldconfig \
    && rm -rf /tmp/cyclonedds

RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git /opt/unitree_sdk2_python \
    && git -C /opt/unitree_sdk2_python checkout ${UNITREE_SDK2_PYTHON_COMMIT} \
    && uv pip install --python "${VIRTUAL_ENV}/bin/python" "cyclonedds==0.10.2" "pyzmq>=25" \
    && uv pip install --python "${VIRTUAL_ENV}/bin/python" --no-deps -e /opt/unitree_sdk2_python \
    && "${VIRTUAL_ENV}/bin/python" -c "import cyclonedds, zmq, unitree_sdk2py; from unitree_sdk2py.utils.crc import CRC; CRC(); print('unitree_sdk2py OK')"

COPY tools/build_r1_brainco_model.py /opt/r1_tools/build_r1_brainco_model.py
COPY r1_sim/r1_sim/joint_map.py /opt/r1_tools/joint_map.py
RUN git clone --filter=blob:none --no-checkout https://github.com/unitreerobotics/unitree_ros.git /tmp/unitree_ros \
    && cd /tmp/unitree_ros \
    && git sparse-checkout init --cone \
    && git sparse-checkout set robots/r1_description robots/g1_with_brainco_hand \
    && git checkout ${UNITREE_ROS_COMMIT} \
    && "${VIRTUAL_ENV}/bin/python" /opt/r1_tools/build_r1_brainco_model.py \
        --r1-dir robots/r1_description \
        --brainco-urdf robots/g1_with_brainco_hand/g1_29dof_mode_15_brainco_hand.urdf \
        --joint-map /opt/r1_tools/joint_map.py \
        --out-dir /opt/r1_model \
        --strip-mimic \
    && (cp LICENSE /opt/r1_model/LICENSE.unitree_ros 2>/dev/null || true) \
    && rm -rf /tmp/unitree_ros \
    && test -f /opt/r1_model/r1_brainco.urdf

COPY --chown=isaaclab:isaaclab r1_sim /workspace/r1_sim
COPY docker/sim/start_sim.sh /usr/local/bin/start_sim.sh
RUN chmod 755 /usr/local/bin/start_sim.sh \
    && mkdir -p /root/.cache/r1_sim/usd \
    && chown -R isaaclab:isaaclab /root/.cache/r1_sim

ENV R1_URDF=/opt/r1_model/r1_brainco.urdf \
    R1_USD_CACHE=/root/.cache/r1_sim/usd \
    DDS_DOMAIN_ID=1 \
    DDS_INTERFACE=lo \
    ACCEPT_EULA=Y \
    OMNI_KIT_ACCEPT_EULA=YES \
    PRIVACY_CONSENT=N

HEALTHCHECK --interval=15s --timeout=5s --start-period=900s --retries=3 \
    CMD bash -c '[ $(( $(date +%s) - $(stat -c %Y /tmp/r1_sim.status) )) -lt 30 ]'

USER isaaclab
WORKDIR /workspace/r1_sim
ENTRYPOINT ["/usr/local/bin/start_sim.sh"]
