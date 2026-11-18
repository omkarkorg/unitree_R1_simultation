from __future__ import annotations

import math

import numpy as np


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_from_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(angle / 2.0)
    return np.array([math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s])


def wxyz_to_xyzw(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([x, y, z, w], dtype=np.float64)


def xyzw_to_wxyz(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([w, x, y, z], dtype=np.float64)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_rpy(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


ROS_OPTICAL_LOOKING_FORWARD = np.array([0.5, -0.5, 0.5, -0.5])


def camera_quat_ros(pitch_down_rad: float) -> np.ndarray:
    tilt = quat_from_axis_angle([0.0, 1.0, 0.0], pitch_down_rad)
    q = quat_mul(tilt, ROS_OPTICAL_LOOKING_FORWARD)
    return q / np.linalg.norm(q)


def horizontal_aperture_for_vfov(vfov_rad: float, focal_length: float, width: int, height: int) -> float:
    vertical_aperture = 2.0 * focal_length * math.tan(vfov_rad / 2.0)
    return vertical_aperture * width / height
