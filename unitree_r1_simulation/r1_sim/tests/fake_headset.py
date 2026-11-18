#!/usr/bin/env python3

import asyncio
import ssl
import threading
import time

import aiohttp
import msgpack
import numpy as np

HEAD = (0.0, 1.6, 0.0)
START_POSE = {"left": (-0.2, 1.15, -0.35), "right": (0.2, 1.15, -0.35)}
WRIST_ROT = {
    "right": np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    "left": np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
}
FINGER_SEGMENTS = (0.04, 0.025, 0.02)
JITTER = 0.0003
CONTROLLER_POSE = {"left": (-0.2, 1.0, -0.3), "right": (0.2, 1.0, -0.3)}


def hand_joints(side, curl):
    s = -1.0 if side == "right" else 1.0
    pts = [np.zeros(3)]
    thumb_open = np.array([[0.025, -0.010, -0.025], [0.045, -0.015, -0.045], [0.060, -0.015, -0.065], [0.070, -0.015, -0.085]])
    thumb_closed = np.array([[0.025, -0.020, -0.025], [0.035, -0.035, -0.045], [0.030, -0.045, -0.065], [0.020, -0.050, -0.080]])
    for p in (1 - curl) * thumb_open + curl * thumb_closed:
        pts.append(np.array([s * p[0], p[1], p[2]]))
    for x, scale in ((0.022, 1.0), (0.0, 1.08), (-0.02, 1.0), (-0.04, 0.82)):
        base = np.array([s * x, 0.0, -0.03])
        knuckle = np.array([s * x, 0.0, -0.09 * scale])
        pts += [base, knuckle]
        pos, angle = knuckle.copy(), 0.0
        for length in FINGER_SEGMENTS:
            angle += curl * np.pi / 2.5
            pos = pos + scale * length * np.array([0.0, -np.sin(angle), -np.cos(angle)])
            pts.append(pos)
    return np.array(pts)


def hand_matrices(side, wrist_pos, curl, rng=np.random.default_rng()):
    rot = WRIST_ROT[side]
    out = []
    for p in hand_joints(side, curl):
        m = np.eye(4)
        m[:3, :3] = rot
        m[:3, 3] = np.asarray(wrist_pos) + rot @ p + rng.normal(0.0, JITTER, 3)
        out += m.flatten(order="F").tolist()
    return out


def head_matrix(yaw_deg, pitch_deg, position=HEAD):
    a, b = np.radians(yaw_deg), np.radians(-pitch_deg)
    ry = np.array([[np.cos(a), 0.0, np.sin(a)], [0.0, 1.0, 0.0], [-np.sin(a), 0.0, np.cos(a)]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(b), -np.sin(b)], [0.0, np.sin(b), np.cos(b)]])
    m = np.eye(4)
    m[:3, :3], m[:3, 3] = ry @ rx, position
    return m


def _without_bytes(node):
    return {k: v for k, v in node.items() if not isinstance(v, (bytes, bytearray))}


class FakeHeadset:
    def __init__(self, url="wss://127.0.0.1:8012/", rate=30.0):
        self.url, self.period = url, 1.0 / rate
        self.sent = []
        self.lock = threading.Lock()
        self.hands = None
        self.head = head_matrix(0.0, 0.0)
        self.controllers = False
        self.last_controllers = None
        self.pending_glitch = None
        self.planes = {}
        self.nodes = {}
        self.parents = {}
        self.connected = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=lambda: asyncio.run(self._run_quietly()), daemon=True)

    def start(self, timeout=15.0):
        self._thread.start()
        if not self.connected.wait(timeout):
            raise RuntimeError(f"could not connect to the teleop web server at {self.url}")
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=5)

    def set_hands(self, left=None, right=None, curl=0.0):
        with self.lock:
            self.hands = (left or START_POSE["left"], right or START_POSE["right"], float(curl))

    def joints(self, side):
        with self.lock:
            hands = self.hands
        if hands is None:
            return None
        wrist = np.asarray(hands[0] if side == "left" else hands[1], dtype=float)
        return wrist + (WRIST_ROT[side] @ hand_joints(side, hands[2]).T).T

    def wrist_for_fingertip(self, side, point, curl=0.0):
        return np.asarray(point, dtype=float) - WRIST_ROT[side] @ hand_joints(side, curl)[9]

    def poke(self, side, point, curl=0.0):
        with self.lock:
            left, right, _ = self.hands or (START_POSE["left"], START_POSE["right"], 0.0)
        wrist = tuple(self.wrist_for_fingertip(side, point, curl))
        self.set_hands(left=wrist if side == "left" else left, right=wrist if side == "right" else right, curl=curl)

    def hide_hands(self):
        with self.lock:
            self.hands = None

    def set_head(self, yaw=0.0, pitch=0.0, position=HEAD):
        with self.lock:
            self.head = head_matrix(yaw, pitch, position)

    def use_controllers(self, on=True):
        with self.lock:
            self.controllers = bool(on)

    def glitch(self, side="right", offset=(0.0, 0.0, 0.3)):
        with self.lock:
            self.pending_glitch = (side, np.asarray(offset, dtype=float))

    def sent_at(self, t):
        return next((s for s in self.sent if s >= t), t)

    def image_planes(self):
        with self.lock:
            return {k: dict(v) for k, v in self.planes.items()}

    def scene(self):
        with self.lock:
            return {k: dict(v) for k, v in self.nodes.items()}, dict(self.parents)

    def _frame(self, rng):
        ts = int(time.time() * 1000)
        with self.lock:
            hands, head, controllers, glitch = self.hands, self.head.copy(), self.controllers, self.pending_glitch
            hands_tracked = hands is not None and not controllers
            if hands_tracked:
                self.pending_glitch = None
        msgs = [{"etype": "CAMERA_MOVE", "key": "defaultCamera", "ts": ts,
                 "value": {"camera": {"matrix": head.flatten(order="F").tolist()}}}]
        value = {"leftState": {}, "rightState": {}}
        if hands_tracked:
            left, right, curl = hands
            pos = {"left": np.asarray(left, dtype=float), "right": np.asarray(right, dtype=float)}
            if glitch is not None:
                pos[glitch[0]] = pos[glitch[0]] + glitch[1]
            state = {"pinch": False, "pinchValue": 0.08, "squeeze": False, "squeezeValue": 0.1}
            value = {"left": hand_matrices("left", pos["left"], curl, rng),
                     "right": hand_matrices("right", pos["right"], curl, rng), "leftState": state, "rightState": state}
        hand_msg = {"etype": "HAND_MOVE", "key": "hands", "ts": ts, "value": value}
        if controllers or hands_tracked:
            poses = {}
            for side in ("left", "right"):
                m = np.eye(4)
                origin = CONTROLLER_POSE[side] if controllers else pos[side] + [0.0, 0.0, -0.05]
                m[:3, 3] = np.asarray(origin) + rng.normal(0.0, 0.002, 3)
                poses[side] = m.flatten(order="F").tolist()
            self.last_controllers = poses
        ctrl = {} if self.last_controllers is None else {**self.last_controllers, "leftState": {}, "rightState": {}}
        msgs += [{"etype": "CONTROLLER_MOVE", "key": "motionControllers", "ts": ts, "value": ctrl}, hand_msg]
        return msgs

    async def _run(self):
        ctx = ssl.create_default_context()
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
        rng = np.random.default_rng()
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.url, ssl=ctx, max_msg_size=0, heartbeat=10) as ws:
                self.connected.set()
                drain = asyncio.create_task(self._drain(ws))
                while not self._stop.is_set() and not ws.closed:
                    for msg in self._frame(rng):
                        await ws.send_bytes(msgpack.packb(msg, use_single_float=True))
                    self.sent = self.sent[-1999:] + [time.time()]
                    await asyncio.sleep(self.period)
                drain.cancel()

    async def _run_quietly(self):
        try:
            await self._run()
        except (aiohttp.ClientError, ConnectionError, OSError):
            pass

    async def _drain(self, ws):
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.BINARY:
                continue
            try:
                event = msgpack.unpackb(msg.data, raw=False)
            except (ValueError, msgpack.ExtraData):
                continue
            if not isinstance(event, dict):
                continue
            etype, data, now = event.get("etype"), event.get("data") or {}, time.time()
            with self.lock:
                if etype in ("UPSERT", "ADD"):
                    parent = data.get("to") if etype == "ADD" else None
                    stack = [(n, parent) for n in (data.get("nodes") or [])]
                    while stack:
                        node, parent = stack.pop()
                        if isinstance(node, list):
                            stack.extend((n, parent) for n in node)
                            continue
                        if not isinstance(node, dict) or "key" not in node:
                            continue
                        key = node["key"]
                        self.nodes[key] = {**_without_bytes({k: v for k, v in node.items() if k != "children"}), "received": now}
                        self.parents[key] = parent
                        stack.extend((c, key) for c in node.get("children") or [])
                        if str(key).startswith("background"):
                            self.planes[key] = {**_without_bytes(node), "received": now}
                elif etype == "UPDATE":
                    for node in data.get("nodes") or []:
                        if isinstance(node, dict) and node.get("key") in self.nodes:
                            self.nodes[node["key"]].update({**_without_bytes(node), "received": now})
                elif etype == "REMOVE":
                    for key in data.get("keys") or []:
                        self.nodes.pop(key, None)
                        self.parents.pop(key, None)
