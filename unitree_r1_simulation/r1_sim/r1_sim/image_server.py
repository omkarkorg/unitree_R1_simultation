from __future__ import annotations

import io
import json
import threading

import numpy as np
import zmq
from PIL import Image

CONFIG_PORT = 60000
HEAD_ZMQ_PORT = 55555
LEFT_WRIST_ZMQ_PORT = 55556
RIGHT_WRIST_ZMQ_PORT = 55557


def make_camera_config(image_shape: tuple[int, int], binocular: bool, fps: int,
                       head_port: int = HEAD_ZMQ_PORT, geometry: dict | None = None, extra: dict | None = None) -> dict:

    def disabled(zmq_port: int, webrtc_port: int) -> dict:
        return {
            "enable_zmq": False,
            "zmq_port": zmq_port,
            "enable_webrtc": False,
            "webrtc_port": webrtc_port,
            "webrtc_codec": "h264",
            "type": "isaacsim",
            "image_shape": [480, 640],
            "binocular": False,
            "fps": fps,
        }

    return {
        "head_camera": {
            "enable_zmq": True,
            "zmq_port": head_port,
            "enable_webrtc": False,
            "webrtc_port": 60001,
            "webrtc_codec": "h264",
            "type": "isaacsim",
            "image_shape": [int(image_shape[0]), int(image_shape[1])],
            "binocular": bool(binocular),
            "fps": int(fps),
            **(geometry or {}),
        },
        "left_wrist_camera": disabled(LEFT_WRIST_ZMQ_PORT, 60002),
        "right_wrist_camera": disabled(RIGHT_WRIST_ZMQ_PORT, 60003),
        **(extra or {}),
    }


class TeleimagerServer:
    def __init__(self, image_shape: tuple[int, int], binocular: bool, fps: int = 30,
                 bind_host: str = "0.0.0.0", config_port: int = CONFIG_PORT,
                 head_port: int = HEAD_ZMQ_PORT, jpeg_quality: int = 85, geometry: dict | None = None,
                 extra: dict | None = None):
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self.config = make_camera_config(self.image_shape, binocular, fps, head_port, geometry, extra)
        self._jpeg_quality = int(jpeg_quality)
        self._ctx = zmq.Context()
        self._running = True
        self._frame: tuple[np.ndarray, dict | None] | None = None
        self._frame_cv = threading.Condition()
        self.frames_sent = 0
        self.config_requests = 0

        self._rep = self._ctx.socket(zmq.REP)
        self._rep.setsockopt(zmq.LINGER, 0)
        self._rep.bind(f"tcp://{bind_host}:{config_port}")

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.LINGER, 0)
        self._pub.setsockopt(zmq.SNDHWM, 1)
        self._pub.setsockopt(zmq.CONFLATE, 1)
        self._pub.bind(f"tcp://{bind_host}:{head_port}")

        self._threads = [
            threading.Thread(target=self._config_loop, name="teleimager-config", daemon=True),
            threading.Thread(target=self._publish_loop, name="teleimager-publish", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def submit_frame(self, rgb: np.ndarray, meta: dict | None = None) -> None:
        if rgb.shape != (*self.image_shape, 3):
            raise ValueError(f"frame shape {rgb.shape} != configured {(*self.image_shape, 3)}")
        with self._frame_cv:
            self._frame = (rgb, meta)
            self._frame_cv.notify()

    def _config_loop(self) -> None:
        poller = zmq.Poller()
        poller.register(self._rep, zmq.POLLIN)
        while self._running:
            try:
                events = dict(poller.poll(100))
                if self._rep in events:
                    self._rep.recv()
                    self._rep.send_json(self.config)
                    self.config_requests += 1
            except zmq.ZMQError:
                if self._running:
                    raise
                return

    def _publish_loop(self) -> None:
        while self._running:
            with self._frame_cv:
                while self._running and self._frame is None:
                    self._frame_cv.wait(timeout=0.1)
                item, self._frame = self._frame, None
            if item is None:
                continue
            frame, meta = item
            buf = io.BytesIO()
            Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8), "RGB").save(
                buf, format="JPEG", quality=self._jpeg_quality)
            jpg = buf.getvalue()
            if meta:
                comment = json.dumps(meta, separators=(",", ":")).encode()
                jpg = jpg[:2] + b"\xff\xfe" + (len(comment) + 2).to_bytes(2, "big") + comment + jpg[2:]
            try:
                self._pub.send(jpg, zmq.NOBLOCK)
                self.frames_sent += 1
            except zmq.Again:
                pass
            except zmq.ZMQError:
                if self._running:
                    raise
                return

    def close(self) -> None:
        self._running = False
        with self._frame_cv:
            self._frame_cv.notify_all()
        for t in self._threads:
            t.join(timeout=1.0)
        self._rep.close(0)
        self._pub.close(0)
        self._ctx.term()
