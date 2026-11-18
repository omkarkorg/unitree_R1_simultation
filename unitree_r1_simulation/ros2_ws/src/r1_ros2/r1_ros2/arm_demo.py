import math
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from unitree_go.msg import MotorCmd as GoMotorCmd
from unitree_go.msg import MotorCmds
from unitree_hg.msg import LowCmd, LowState

from .crc import hg_lowcmd_crc
from .joints import BODY_JOINT_SLOTS, HAND_NUM_MOTORS, HAND_SIDES, NUM_MOTOR_SLOTS

ARM_SLOTS = [BODY_JOINT_SLOTS[n] for n in BODY_JOINT_SLOTS if "shoulder" in n or "elbow" in n or "wrist" in n]
HEAD_SLOTS = [BODY_JOINT_SLOTS["head_pitch_joint"], BODY_JOINT_SLOTS["head_yaw_joint"]]
GAINS = {"leg_waist": (200.0, 3.0), "arm": (50.0, 2.0), "wrist": (30.0, 2.0), "head": (15.0, 1.0)}
HAND_OPEN = [0.0] * HAND_NUM_MOTORS
HAND_CLOSED = [0.6, 0.6, 0.9, 0.9, 0.9, 0.9]


class ArmDemo(Node):
    def __init__(self):
        super().__init__("r1_arm_demo")
        allow_real = bool(self.declare_parameter("allow_real_robot", False).value)
        self._rate = float(self.declare_parameter("rate", 100.0).value)
        self._amp = float(self.declare_parameter("amplitude", 0.35).value)
        self._period = float(self.declare_parameter("period", 4.0).value)
        domain = int(os.environ.get("ROS_DOMAIN_ID", "0"))
        if domain != 1 and not allow_real:
            raise RuntimeError(f"ROS_DOMAIN_ID={domain}: this demo only runs on the simulation domain 1 "
                               "(pass -p allow_real_robot:=true if you really mean the real robot)")

        qos_state = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(LowState, "lowstate", self._on_state, qos_state)
        self._pub = self.create_publisher(LowCmd, "lowcmd", 10)
        self._hand_pub = {s: self.create_publisher(MotorCmds, f"brainco/{s}/cmd", 10) for s in HAND_SIDES}
        self._state = None
        self._start_q = None
        self._t = 0.0
        self.create_timer(1.0 / self._rate, self._step)
        self.get_logger().info("waiting for rt/lowstate ...")

    def _on_state(self, msg: LowState):
        self._state = msg

    @staticmethod
    def _gains(slot):
        if slot in HEAD_SLOTS:
            return GAINS["head"]
        if slot in (BODY_JOINT_SLOTS["left_wrist_roll_joint"], BODY_JOINT_SLOTS["right_wrist_roll_joint"]):
            return GAINS["wrist"]
        if slot in ARM_SLOTS:
            return GAINS["arm"]
        return GAINS["leg_waist"]

    def _step(self):
        if self._state is None:
            return
        if self._start_q is None:
            self._start_q = [float(self._state.motor_state[i].q) for i in range(NUM_MOTOR_SLOTS)]
            self.get_logger().info("got rt/lowstate - moving arms and head to zero, then waving")
        self._t += 1.0 / self._rate
        ramp = min(1.0, self._t / 3.0)
        wave = 0.0 if self._t < 3.0 else math.sin(2.0 * math.pi * (self._t - 3.0) / self._period)

        targets = {}
        for slot in ARM_SLOTS + HEAD_SLOTS:
            targets[slot] = (1.0 - ramp) * self._start_q[slot]
        if self._t >= 3.0:
            for side, sign in (("left", 1.0), ("right", -1.0)):
                targets[BODY_JOINT_SLOTS[f"{side}_shoulder_pitch_joint"]] = -self._amp * (1.0 - math.cos(
                    2.0 * math.pi * (self._t - 3.0) / self._period))
                targets[BODY_JOINT_SLOTS[f"{side}_elbow_joint"]] = -0.5 * self._amp * (1.0 + wave)
                targets[BODY_JOINT_SLOTS[f"{side}_wrist_roll_joint"]] = sign * 0.5 * wave

        cmd = LowCmd()
        cmd.mode_pr = 0
        cmd.mode_machine = int(self._state.mode_machine)
        used = set(BODY_JOINT_SLOTS.values())
        for slot in range(NUM_MOTOR_SLOTS):
            m = cmd.motor_cmd[slot]
            if slot not in used:
                m.mode = 0
                continue
            kp, kd = self._gains(slot)
            m.mode = 1
            m.q = float(targets.get(slot, self._start_q[slot]))
            m.dq = 0.0
            m.tau = 0.0
            m.kp = kp
            m.kd = kd
        cmd.crc = hg_lowcmd_crc(cmd)
        self._pub.publish(cmd)

        closing = 0.0 if self._t < 3.0 else 0.5 * (1.0 - math.cos(2.0 * math.pi * (self._t - 3.0) / self._period))
        for side in HAND_SIDES:
            hand = MotorCmds()
            hand.cmds = [GoMotorCmd() for _ in range(HAND_NUM_MOTORS)]
            for i, m in enumerate(hand.cmds):
                m.q = HAND_OPEN[i] + closing * (HAND_CLOSED[i] - HAND_OPEN[i])
                m.dq = 1.0
            self._hand_pub[side].publish(hand)


def main(args=None):
    rclpy.init(args=args)
    node = ArmDemo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
