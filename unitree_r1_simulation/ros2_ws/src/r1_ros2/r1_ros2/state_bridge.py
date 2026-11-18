import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from unitree_go.msg import MotorStates
from unitree_hg.msg import LowState

from .joints import (
    BODY_JOINT_SLOTS,
    HAND_COUPLED_JOINTS,
    HAND_NUM_MOTORS,
    HAND_SIDES,
    body_joint_names,
    hand_joint_names,
    hand_motor_to_joint_positions,
)


class StateBridge(Node):
    def __init__(self):
        super().__init__("r1_state_bridge")
        rate = self.declare_parameter("publish_rate", 50.0).value
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self._body_names = body_joint_names()
        self._slots = [BODY_JOINT_SLOTS[n] for n in self._body_names]
        self._hand_names = {s: hand_joint_names(s) for s in HAND_SIDES}
        n_body = len(self._body_names)
        self._q = [0.0] * n_body
        self._dq = [0.0] * n_body
        self._tau = [0.0] * n_body
        self._hand = {s: None for s in HAND_SIDES}
        self._have_body = False
        self._warned = False

        self.create_subscription(LowState, "lowstate", self._on_lowstate, qos)
        for side in HAND_SIDES:
            self.create_subscription(MotorStates, f"brainco/{side}/state",
                                     lambda msg, s=side: self._on_hand(s, msg), qos)
        self._pub = self.create_publisher(JointState, "joint_states", 10)
        self.create_timer(1.0 / rate, self._publish)
        self.get_logger().info("waiting for rt/lowstate ...")

    def _on_lowstate(self, msg: LowState):
        ms = msg.motor_state
        self._q = [float(ms[s].q) for s in self._slots]
        self._dq = [float(ms[s].dq) for s in self._slots]
        self._tau = [float(ms[s].tau_est) for s in self._slots]
        if not self._have_body:
            self.get_logger().info("receiving rt/lowstate")
        self._have_body = True

    def _on_hand(self, side: str, msg: MotorStates):
        if len(msg.states) < HAND_NUM_MOTORS:
            return
        ms = msg.states[:HAND_NUM_MOTORS]
        tau = [float(m.tau_est) for m in ms]
        self._hand[side] = (hand_motor_to_joint_positions([m.q for m in ms]),
                            hand_motor_to_joint_positions([m.dq for m in ms]),
                            tau + [0.0] * len(HAND_COUPLED_JOINTS))

    def _publish(self):
        if not self._have_body:
            return
        names = list(self._body_names)
        pos, vel, eff = list(self._q), list(self._dq), list(self._tau)
        for side in HAND_SIDES:
            hand = self._hand[side]
            if hand is None:
                if not self._warned:
                    self.get_logger().warn(f"no rt/brainco/{side}/state yet - hand joints not published")
                    self._warned = True
                continue
            names += self._hand_names[side]
            pos += hand[0]
            vel += hand[1]
            eff += hand[2]
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name, js.position, js.velocity, js.effort = names, pos, vel, eff
        self._pub.publish(js)


def main(args=None):
    rclpy.init(args=args)
    node = StateBridge()
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
