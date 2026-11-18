from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

CLOSE, OPEN = 0.40, 0.25
POINT, FINGER = 0.06, 0.03
TAU = 0.02
SLIP = 0.06
LEAD = 0.08
SETTLE, SETTLE_MAX = 0.15, 0.03
V_MAX, W_MAX = 3.0, 15.0
FINGERS = ("index", "middle")


def tip_points(urdf_path: str) -> dict:
    points = {}
    for joint in ET.parse(urdf_path).getroot().findall("joint"):
        child = joint.find("child").get("link")
        if joint.get("type") == "fixed" and child.endswith("_tip_link"):
            points[joint.find("parent").get("link")] = np.array([float(v) for v in joint.find("origin").get("xyz").split()])
    return points


def object_box(obj: dict) -> tuple:
    if obj.get("pieces") is not None:
        points = np.vstack([np.asarray(p, dtype=float) for p in obj["pieces"]])
        lo, hi = points.min(axis=0), points.max(axis=0)
        return (lo + hi) / 2.0, (hi - lo) / 2.0
    if obj.get("shape") == "box":
        return np.zeros(3), np.asarray(obj["size"], dtype=float) / 2.0
    return np.zeros(3), np.array([obj["radius"], obj["radius"], obj["height"] / 2.0])


def pose_matrix(pose, origin=(0.0, 0.0, 0.0)) -> np.ndarray:
    x, y, z, w = (float(v) for v in pose[3:7])
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = np.asarray(pose[:3], dtype=float) - np.asarray(origin, dtype=float)
    return T


def _rotation_log(R: np.ndarray) -> np.ndarray:
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s, c = np.linalg.norm(v) / 2.0, (np.trace(R) - 1.0) / 2.0
    return v * (0.5 if s < 1e-9 else math.atan2(s, c) / (2.0 * s))


class GripAssist:
    def __init__(self, boxes: list, dt: float):
        self.boxes, self.dt = boxes, dt
        self.held = {}
        self.nearest = {}

    def clear(self) -> None:
        self.held.clear()

    def holder(self, index: int):
        return next((side for side, h in self.held.items() if h["object"] == index), None)

    def _distance(self, points: np.ndarray, pose: np.ndarray, index: int) -> np.ndarray:
        centre, half = self.boxes[index]
        local = (points - pose[:3, 3]) @ pose[:3, :3] - centre
        return np.linalg.norm(np.maximum(np.abs(local) - half, 0.0), axis=-1)

    def try_grip(self, side: str, wrist: np.ndarray, thumb: np.ndarray, fingers: np.ndarray, closure: float,
                 hand_q: np.ndarray, poses: list):
        if side in self.held or closure < CLOSE:
            return None
        best, best_d = None, np.inf
        point = 0.5 * (thumb + fingers.mean(axis=0))
        for i, pose in enumerate(poses):
            if self.holder(i) is not None:
                continue
            d_point = float(self._distance(point[None], pose, i)[0])
            d_finger = float(self._distance(fingers, pose, i).min())
            if d_point <= POINT and d_finger <= FINGER and d_point < best_d:
                best, best_d = i, d_point
            if d_point + d_finger < sum(self.nearest.get(side, (np.inf, 0.0, 0))[:2]):
                self.nearest[side] = (d_point, d_finger, i)
        if best is not None:
            self.nearest.pop(side, None)
            centre, half = self.boxes[best]
            local = (point - poses[best][:3, 3]) @ poses[best][:3, :3] - centre
            touch = poses[best][:3, :3] @ (np.clip(local, -half, half) - local)
            touch *= min(1.0, SETTLE_MAX / max(np.linalg.norm(touch), 1e-9))
            self.held[side] = {"object": best, "in_hand": np.linalg.inv(wrist) @ poses[best], "last": None, "t": 0.0,
                               "settle": -wrist[:3, :3].T @ touch}
        return best

    def hold(self, side: str, wrist: np.ndarray, closure: float, pose: np.ndarray):
        h = self.held[side]
        h["t"] += self.dt
        in_hand = h["in_hand"].copy()
        in_hand[:3, 3] += min(1.0, h["t"] / SETTLE) * h["settle"]
        want = wrist @ in_hand
        error = want[:3, 3] - pose[:3, 3]
        if closure < OPEN or np.linalg.norm(error) > SLIP:
            del self.held[side]
            return None
        along = np.zeros(3) if h["last"] is None else (want[:3, 3] - h["last"][:3, 3]) / self.dt
        h["last"] = want
        v = along + error / TAU + np.array([0.0, 0.0, 9.81 * self.dt])
        w = _rotation_log(want[:3, :3] @ pose[:3, :3].T) / TAU
        return v * min(1.0, V_MAX / max(np.linalg.norm(v), 1e-9)), w * min(1.0, W_MAX / max(np.linalg.norm(w), 1e-9))

    @staticmethod
    def finger_limit(hand_q: np.ndarray) -> np.ndarray:
        return hand_q + LEAD
