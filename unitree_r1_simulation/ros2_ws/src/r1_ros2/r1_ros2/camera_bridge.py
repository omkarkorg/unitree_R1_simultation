from array import array

import cv2
import numpy as np
import rclpy
import zmq
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image


class CameraBridge(Node):
    def __init__(self):
        super().__init__("r1_camera_bridge")
        host = self.declare_parameter("host", "127.0.0.1").value
        port = int(self.declare_parameter("port", 55555).value)
        self._frame_id = self.declare_parameter("frame_id", "head_camera").value
        self._publish_raw = bool(self.declare_parameter("publish_raw", True).value)

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.RCVHWM, 1)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(f"tcp://{host}:{port}")
        self._sock.setsockopt_string(zmq.SUBSCRIBE, "")

        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self._pub_jpeg = self.create_publisher(CompressedImage, "r1/head_camera/image_raw/compressed", qos)
        self._pub_raw = self.create_publisher(Image, "r1/head_camera/image_raw", qos) if self._publish_raw else None
        self._frames = 0
        self.create_timer(1.0 / 60.0, self._poll)
        self.get_logger().info(f"subscribed to tcp://{host}:{port}")

    def _poll(self):
        try:
            data = self._sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            return
        stamp = self.get_clock().now().to_msg()
        jpeg = CompressedImage()
        jpeg.header.stamp = stamp
        jpeg.header.frame_id = self._frame_id
        jpeg.format = "jpeg"
        jpeg.data = array("B", data)
        self._pub_jpeg.publish(jpeg)
        if self._pub_raw is not None:
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                msg = Image()
                msg.header = jpeg.header
                msg.height, msg.width = int(img.shape[0]), int(img.shape[1])
                msg.encoding = "bgr8"
                msg.is_bigendian = 0
                msg.step = int(img.shape[1] * 3)
                msg.data = array("B", img.tobytes())
                self._pub_raw.publish(msg)
        self._frames += 1
        if self._frames == 1:
            self.get_logger().info("receiving camera frames")

    def destroy_node(self):
        self._sock.close(0)
        self._ctx.term()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraBridge()
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
