from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass

import numpy as np

from .joint_map import (
    HAND_MOTOR_RANGES,
    HAND_NUM_MOTORS,
    HAND_SIDES,
    NUM_MOTOR_SLOTS,
    R1_BODY_JOINT_SLOTS,
    hand_joint_names,
    hand_motor_to_joint_positions,
)

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_RESET = "rt/reset_pose/cmd"
TOPIC_SIM_STATE = "rt/sim_state"
TOPIC_ARM_SDK = "rt/arm_sdk"
TOPIC_BODY_CMD = "rt/sim/body_cmd"

HAND_DRIVE_KP = 3.0
HAND_DRIVE_KD = 0.1
HAND_NUM_JOINTS = len(hand_joint_names(HAND_SIDES[0]))


def topic_hand_cmd(side: str) -> str:
    return f"rt/brainco/{side}/cmd"


def topic_hand_state(side: str) -> str:
    return f"rt/brainco/{side}/state"


@dataclass
class MotorCommand:
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    enabled: np.ndarray
    stamp: float
    weight: float = 1.0


class R1DdsBridge:
    def __init__(
        self,
        body_joint_names: list[str],
        domain_id: int = 1,
        interface: str | None = "lo",
        enable_hands: bool = True,
        mode_machine: int = 0,
    ):
        from cyclonedds.domain import DomainParticipant
        from cyclonedds.qos import Policy, Qos
        from cyclonedds.sub import DataReader
        from cyclonedds.topic import Topic
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorState_, unitree_hg_msg_dds__LowState_
        from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        self._LowCmd, self._HandCmd = LowCmd_, MotorCmds_

        unknown = [n for n in body_joint_names if n not in R1_BODY_JOINT_SLOTS]
        if unknown:
            raise ValueError(f"joints without a DDS slot: {unknown}")
        self._body_slots = np.array([R1_BODY_JOINT_SLOTS[n] for n in body_joint_names], dtype=np.int64)
        self._n_body = len(body_joint_names)
        self._enable_hands = enable_hands
        self._mode_machine = int(mode_machine)
        self._lock = threading.Lock()
        self._crc = CRC()

        self._lowcmd: MotorCommand | None = None
        self._arm_sdk: MotorCommand | None = None
        self._body_cmd: tuple[list[float], float] | None = None
        self._hand_cmd: dict[str, MotorCommand | None] = {s: None for s in HAND_SIDES}
        self._reset_requests: list[str] = []
        self.stats = {"lowcmd": 0, "lowcmd_crc_errors": 0, "hand_cmd": 0, "hand_cmd_invalid": 0, "reset": 0,
                      "arm_sdk": 0, "body_cmd": 0, "body_cmd_invalid": 0}

        ChannelFactoryInitialize(domain_id, interface if interface else None)
        self._participant = DomainParticipant(domain_id)
        latest_only = Qos(Policy.History.KeepLast(1), Policy.Reliability.BestEffort)

        def latest_reader(name, idl_type):
            return DataReader(self._participant, Topic(self._participant, name, idl_type), qos=latest_only)

        self._lowstate_msg = unitree_hg_msg_dds__LowState_()
        self._lowstate_msg.mode_pr = 0
        self._lowstate_msg.mode_machine = self._mode_machine
        for slot in self._body_slots:
            self._lowstate_msg.motor_state[int(slot)].mode = 1
        self._lowstate_pub = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self._lowstate_pub.Init()
        self._lowcmd_reader = latest_reader(TOPIC_LOWCMD, LowCmd_)
        self._arm_sdk_reader = latest_reader(TOPIC_ARM_SDK, LowCmd_)

        self._reset_sub = ChannelSubscriber(TOPIC_RESET, String_)
        self._reset_sub.Init(self._on_reset, 10)
        self._String = String_
        self._body_cmd_reader = latest_reader(TOPIC_BODY_CMD, String_)
        self._sim_state_pub = ChannelPublisher(TOPIC_SIM_STATE, String_)
        self._sim_state_pub.Init()

        self._hand_state_msg = {}
        self._hand_state_pub = {}
        self._hand_cmd_reader = {}
        if enable_hands:
            for side in HAND_SIDES:
                msg = MotorStates_()
                msg.states = [unitree_go_msg_dds__MotorState_() for _ in range(HAND_NUM_MOTORS)]
                self._hand_state_msg[side] = msg
                pub = ChannelPublisher(topic_hand_state(side), MotorStates_)
                pub.Init()
                self._hand_state_pub[side] = pub
                self._hand_cmd_reader[side] = latest_reader(topic_hand_cmd(side), MotorCmds_)

    @staticmethod
    def _take_latest(reader, idl_type):
        for sample in reversed(reader.take(N=1)):
            if isinstance(sample, idl_type):
                return sample
        return None

    def _parse_lowcmd(self, msg, arm_sdk: bool = False) -> None:
        if self._crc.Crc(msg) != msg.crc:
            self.stats["lowcmd_crc_errors"] += 1
            return
        raw = np.empty((5, NUM_MOTOR_SLOTS), dtype=np.float64)
        mode = np.empty(NUM_MOTOR_SLOTS, dtype=np.int64)
        for i, c in enumerate(msg.motor_cmd[:NUM_MOTOR_SLOTS]):
            raw[0, i] = c.q
            raw[1, i] = c.dq
            raw[2, i] = c.tau
            raw[3, i] = c.kp
            raw[4, i] = c.kd
            mode[i] = c.mode
        s = self._body_slots
        cmd = MotorCommand(
            q=raw[0, s].copy(),
            dq=raw[1, s].copy(),
            tau=raw[2, s].copy(),
            kp=raw[3, s].copy(),
            kd=raw[4, s].copy(),
            enabled=mode[s] != 0,
            stamp=time.monotonic(),
            weight=min(max(float(msg.mode_pr) / 100.0, 0.0), 1.0) if arm_sdk else 1.0,
        )
        if arm_sdk:
            self._arm_sdk = cmd
            self.stats["arm_sdk"] += 1
        else:
            self._lowcmd = cmd
            self.stats["lowcmd"] += 1

    def _parse_body_cmd(self, msg) -> None:
        try:
            d = json.loads(str(msg.data))
            keys = ("x", "y", "yaw", "height", "pitch") if "x" in d else ("vx", "vy", "wz", "height", "pitch")
            cmd = {k: float(d[k]) for k in keys}
        except (ValueError, KeyError, TypeError, AttributeError):
            self.stats["body_cmd_invalid"] += 1
            return
        if not np.all(np.isfinite(list(cmd.values()))):
            self.stats["body_cmd_invalid"] += 1
            return
        self._body_cmd = (cmd, time.monotonic())
        self.stats["body_cmd"] += 1

    def _parse_hand_cmd(self, side: str, msg) -> None:
        cmds = msg.cmds
        q_norm = [float(c.q) for c in cmds[:HAND_NUM_MOTORS]]
        if len(q_norm) < HAND_NUM_MOTORS or not np.all(np.isfinite(q_norm)):
            self.stats["hand_cmd_invalid"] += 1
            return
        n = HAND_NUM_JOINTS
        self._hand_cmd[side] = MotorCommand(
            q=np.array(hand_motor_to_joint_positions(q_norm)),
            dq=np.zeros(n),
            tau=np.zeros(n),
            kp=np.full(n, HAND_DRIVE_KP),
            kd=np.full(n, HAND_DRIVE_KD),
            enabled=np.ones(n, dtype=bool),
            stamp=time.monotonic(),
        )
        self.stats["hand_cmd"] += 1

    def _on_reset(self, msg) -> None:
        with self._lock:
            self._reset_requests.append(str(msg.data).strip())
            self.stats["reset"] += 1

    def get_body_command(self) -> MotorCommand | None:
        msg = self._take_latest(self._lowcmd_reader, self._LowCmd)
        if msg is not None:
            self._parse_lowcmd(msg)
        return self._lowcmd

    def get_arm_sdk_command(self) -> MotorCommand | None:
        msg = self._take_latest(self._arm_sdk_reader, self._LowCmd)
        if msg is not None:
            self._parse_lowcmd(msg, arm_sdk=True)
        return self._arm_sdk

    def get_body_target(self) -> tuple[dict, float] | None:
        msg = self._take_latest(self._body_cmd_reader, self._String)
        if msg is not None:
            self._parse_body_cmd(msg)
        return self._body_cmd

    def get_body_motion_command(self) -> tuple[list[float], float] | None:
        got = self.get_body_target()
        if got is None or "vx" not in got[0]:
            return None
        return [got[0][k] for k in ("vx", "vy", "wz", "height", "pitch")], got[1]

    def get_hand_command(self, side: str) -> MotorCommand | None:
        if not self._enable_hands:
            return None
        msg = self._take_latest(self._hand_cmd_reader[side], self._HandCmd)
        if msg is not None:
            self._parse_hand_cmd(side, msg)
        return self._hand_cmd[side]

    def peek_body_command(self) -> MotorCommand | None:
        return self._lowcmd

    def pop_reset_requests(self) -> list[str]:
        with self._lock:
            requests, self._reset_requests = self._reset_requests, []
        return requests

    def publish_body_state(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau: np.ndarray,
        quat_wxyz: np.ndarray,
        gyro: np.ndarray,
        accel: np.ndarray,
        rpy: np.ndarray,
        sim_time_s: float,
    ) -> None:
        msg = self._lowstate_msg
        for i, slot in enumerate(self._body_slots):
            ms = msg.motor_state[int(slot)]
            ms.q = float(q[i])
            ms.dq = float(dq[i])
            ms.tau_est = float(tau[i])
        imu = msg.imu_state
        imu.quaternion = [float(v) for v in quat_wxyz]
        imu.gyroscope = [float(v) for v in gyro]
        imu.accelerometer = [float(v) for v in accel]
        imu.rpy = [float(v) for v in rpy]
        msg.tick = int(sim_time_s * 1000.0) & 0xFFFFFFFF
        msg.crc = self._crc.Crc(msg)
        self._lowstate_pub.Write(msg)

    def publish_sim_state(self, state: dict) -> None:
        self._sim_state_pub.Write(self._String(data=json.dumps(state, separators=(",", ":"))))

    def publish_hand_state(self, side: str, q: np.ndarray, dq: np.ndarray, tau: np.ndarray) -> None:
        if not self._enable_hands:
            return
        msg = self._hand_state_msg[side]
        for i in range(HAND_NUM_MOTORS):
            ms = msg.states[i]
            ms.q = float(q[i]) / HAND_MOTOR_RANGES[i]
            ms.dq = float(dq[i]) / HAND_MOTOR_RANGES[i]
            ms.tau_est = float(tau[i])
        self._hand_state_pub[side].Write(msg)

    def close(self) -> None:
        try:
            self._reset_sub.Close()
        except Exception:
            pass
        for pub in [self._lowstate_pub, self._sim_state_pub, *self._hand_state_pub.values()]:
            try:
                pub.Close()
            except Exception:
                pass


class JointServo:
    def __init__(self, hold_q: np.ndarray, default_kp: np.ndarray, default_kd: np.ndarray, effort_limit: np.ndarray):
        n = len(hold_q)
        self.n = n
        self.q_des = np.asarray(hold_q, dtype=np.float64).copy()
        self.dq_des = np.zeros(n)
        self.tau_ff = np.zeros(n)
        self.kp = np.asarray(default_kp, dtype=np.float64).copy()
        self.kd = np.asarray(default_kd, dtype=np.float64).copy()
        self.effort_limit = np.asarray(effort_limit, dtype=np.float64).copy()
        self._default = (self.q_des.copy(), self.kp.copy(), self.kd.copy())
        self.last_stamp: float | None = None

    @property
    def hold(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._default

    def reset(self, hold_q: np.ndarray | None = None) -> None:
        q, kp, kd = self._default
        self.q_des = (q if hold_q is None else np.asarray(hold_q, dtype=np.float64)).copy()
        self.dq_des[:] = 0.0
        self.tau_ff[:] = 0.0
        self.kp, self.kd = kp.copy(), kd.copy()
        self.last_stamp = None

    def apply(self, cmd: MotorCommand | None) -> bool:
        if cmd is None or cmd.stamp == self.last_stamp:
            return False
        on = cmd.enabled
        self.q_des = np.where(on, cmd.q, self.q_des)
        self.dq_des = np.where(on, cmd.dq, 0.0)
        self.tau_ff = np.where(on, cmd.tau, 0.0)
        self.kp = np.where(on, np.maximum(cmd.kp, 0.0), 0.0)
        self.kd = np.where(on, np.maximum(cmd.kd, 0.0), 0.0)
        self.last_stamp = cmd.stamp
        return True

    def torque(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        tau = self.kp * (self.q_des - q) + self.kd * (self.dq_des - dq) + self.tau_ff
        return np.clip(tau, -self.effort_limit, self.effort_limit)
