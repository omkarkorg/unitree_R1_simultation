from __future__ import annotations

import json

import zmq

from .layout import LAB_STATE_PORT


class LabStatePublisher:
    def __init__(self, bind_host: str = "0.0.0.0", port: int = LAB_STATE_PORT):
        self._ctx = zmq.Context()
        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.LINGER, 0)
        self._pub.setsockopt(zmq.SNDHWM, 1)
        self._pub.setsockopt(zmq.CONFLATE, 1)
        self._pub.bind(f"tcp://{bind_host}:{port}")
        self.sent = 0

    def publish(self, sim_time: float, q, objects: dict, root=None, stance=None) -> None:
        msg = {"t": round(float(sim_time), 4), "q": [round(float(v), 5) for v in q],
               "objects": {k: [round(float(v), 5) for v in pose] for k, pose in objects.items()}}
        if root is not None:
            msg["root"] = [round(float(v), 5) for v in root]
        if stance is not None:
            msg["stance"] = [round(float(v), 5) for v in stance]
        try:
            self._pub.send(json.dumps(msg, separators=(",", ":")).encode(), zmq.NOBLOCK)
            self.sent += 1
        except zmq.Again:
            pass

    def close(self) -> None:
        self._pub.close(0)
        self._ctx.term()
