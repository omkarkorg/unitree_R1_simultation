#!/usr/bin/env python3

import argparse
import json
import sys
import threading
import time

import numpy as np
import zmq

sys.path.insert(0, "/opt/r1_sim/tests")
sys.path.insert(0, "/opt/r1_sim")
from fake_headset import HEAD, START_POSE, FakeHeadset, head_matrix
from r1_sim import loco_spec
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

ARM_SLOTS = (15, 16, 17, 18, 19, 22, 23, 24, 25, 26)
HEAD_SLOTS = (29, 30)
UPSTREAM_IK_POINT = np.array([0.20, 0.0, 0.0])
R1_MODEL_URDF = "/opt/r1_model/r1_brainco.urdf"
SPEED_TOLERANCE = 1.35
TABLE = (0.0, 45.0)
LOOK = (30.0, 35.0)
NECK_TO_EYE = np.array([0.0, 0.075, -0.0805])
CAM_PITCH, CAM_VFOV = np.radians(50.0), 70.0
HEAD_SPEED_LIMIT = 3.0
R_OPENXR_ROBOT = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
EXACT_START = {"left": (-0.14, 1.20, -0.15), "right": (0.14, 1.20, -0.15)}
EXACT_MOVE = np.array([0.0, 0.06, 0.06])
MOVE = np.array([0.0, 0.08, 0.08])
COLOCATION_TOLERANCE = {"mock": 0.015, "isaac": 0.03}
TRACE_TOLERANCE = {"mock": 0.04, "isaac": 0.05}
TRACES = ((0.05, 2.0), (0.08, 1.2))
STEP_SIZE, STEP_TIME = 0.04, 0.15


class Recorder:
    def __init__(self, ee_offset):
        self.lock = threading.Lock()
        self.ee_offset = ee_offset
        self.data = {"cmd": [], "head": [], "dq": [], "q": [], "hand": [], "ee": [], "head_link": [], "wrist_pose": [],
                     "body": [], "root": [], "reset": []}

    def _add(self, key, value):
        with self.lock:
            self.data[key].append((time.time(), value))

    def on_lowcmd(self, msg):
        self._add("cmd", [msg.motor_cmd[i].q for i in ARM_SLOTS])
        self._add("head", [msg.motor_cmd[i].q for i in HEAD_SLOTS])

    def on_lowstate(self, msg):
        self._add("dq", max(abs(msg.motor_state[i].dq) for i in ARM_SLOTS))
        self._add("q", [msg.motor_state[i].q for i in ARM_SLOTS])

    def on_hand(self, msg):
        self._add("hand", [s.q for s in msg.states])

    def on_body_cmd(self, msg):
        d = json.loads(msg.data)
        keys = ("x", "y", "yaw") if "x" in d else ("vx", "vy", "wz")
        self._add("body", [d[keys[0]], d[keys[1]], d[keys[2]], d["height"], d["pitch"]])

    def on_reset(self, msg):
        self._add("reset", [float(msg.data) if msg.data.strip().isdigit() else -1.0])

    def on_sim_state(self, msg):
        state = json.loads(msg.data)
        if state.get("robot"):
            r = state["robot"]
            R = quat_matrix(r[3:])
            self._add("root", [r[0], r[1], r[2], float(np.arctan2(R[1, 0], R[0, 0]))])
        wrist = state.get("wrists", {}).get("right")
        if wrist:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = quat_matrix(wrist[3:]), wrist[:3]
            self._add("ee", list(T[:3, 3] + T[:3, :3] @ self.ee_offset["right"]))
            self._add("wrist_pose", T.flatten().tolist())
        if state.get("head"):
            self._add("head_link", list(quat_matrix(state["head"][3:])[:, 0]))

    def get(self, key):
        with self.lock:
            rows = list(self.data[key])
        if not rows:
            return np.zeros(0), np.zeros((0, 1))
        return np.array([r[0] for r in rows]), np.array([r[1] for r in rows], dtype=float)


def quat_matrix(q_xyzw):
    x, y, z, w = q_xyzw
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def palm_offsets():
    import pinocchio as pin
    model = pin.buildModelFromUrdf(R1_MODEL_URDF)
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    out = {}
    for side in ("left", "right"):
        W = data.oMf[model.getFrameId(f"{side}_wrist_roll_link")]
        mid = 0.5 * (data.oMf[model.getFrameId(f"{side}_hand_palm_link")].translation
                     + data.oMf[model.getFrameId(f"{side}_middle_proximal_link")].translation)
        out[side] = W.rotation.T @ (mid - W.translation)
    return out


def head_link_rotation(head_q):
    p, y = head_q
    rot_y = np.array([[np.cos(p), 0.0, np.sin(p)], [0.0, 1.0, 0.0], [-np.sin(p), 0.0, np.cos(p)]])
    rot_z = np.array([[np.cos(y), -np.sin(y), 0.0], [np.sin(y), np.cos(y), 0.0], [0.0, 0.0, 1.0]])
    return rot_y @ rot_z


def camera_forward(head_q):
    return head_link_rotation(head_q) @ np.array([np.cos(CAM_PITCH), 0.0, -np.sin(CAM_PITCH)])


def gaze_robot(yaw_deg, pitch_deg):
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    return np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), -np.sin(p)])


def angle_deg(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1.0, 1.0))))


def ipc(cmd):
    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect("ipc://@xr_teleoperate_data.ipc")
    sock.send_json({"reqid": "teleop_hand_test", "cmd": cmd})
    reply = sock.recv_json() if sock.poll(3000) else None
    sock.close()
    return reply


def max_mean_speed(t, q, span=0.1):
    speeds = [0.0]
    for k in range(len(t)):
        j = np.searchsorted(t, t[k] + span)
        if j < len(t):
            speeds.append(np.max(np.abs(q[j] - q[k])) / (t[j] - t[k]))
    return max(speeds)


def max_speed_at(t, q, span=0.03):
    best = (0.0, None)
    for k in range(len(t)):
        j = np.searchsorted(t, t[k] + span)
        if j < len(t):
            best = max(best, (float(np.max(np.abs(q[j] - q[k])) / (t[j] - t[k])), float(t[k])), key=lambda b: b[0])
    return best


def window(t, v, t0, t1):
    return v[(t >= t0) & (t <= t1)]


def mean_change(t, v, before, after, span=0.4):
    return window(t, v, after - span, after).mean(axis=0) - window(t, v, before - span, before).mean(axis=0)


def matrix(node):
    return np.array(node["matrix"], dtype=float).reshape(4, 4, order="F")


def operator_palm_centre(points):
    return 0.5 * (points[0] + points[11])


def drawn_palm_centre(nodes, side):
    lab = matrix(nodes["lab"])
    pts = [lab @ matrix(nodes[f"link-{side}_{link}"])[:, 3] for link in ("hand_palm_link", "middle_proximal_link")]
    return 0.5 * (pts[0][:3] + pts[1][:3])


def body_motion(cmds):
    if not len(cmds):
        return 0.0
    if POSE_COMMANDS[0]:
        return float(np.abs(cmds[:, :3] - cmds[0, :3]).max())
    return float(np.abs(cmds[:, :3]).max())


def body_still(cmds):
    return body_motion(cmds) < 1e-6


POSE_COMMANDS = [False]


def eye_in_pelvis(mount=(0.095, 0.0, 0.0)):
    import pinocchio as pin
    model = pin.buildModelFromUrdf(R1_MODEL_URDF)
    data = model.createData()
    pin.framesForwardKinematics(model, data, pin.neutral(model))
    return np.asarray(data.oMf[model.getFrameId("head_yaw_link")].act(np.array(mount, dtype=float)))


def drawn_eye(nodes, eye):
    return (matrix(nodes["lab"]) @ matrix(nodes["link-pelvis_link"]) @ np.append(eye, 1.0))[:3]


def check_view(planes, failures, info):
    keys = sorted(k for k in planes if k.startswith("background"))
    if not keys:
        failures.append("the headset received no head-camera image")
        return
    stereo = "background-left" in keys
    distance = 2.0 if stereo else 1.0
    gaze = R_OPENXR_ROBOT @ gaze_robot(*LOOK)
    for key in keys:
        p = planes[key]
        if not p.get("fixed") or not p.get("matrix"):
            failures.append(f"{key}: the image is not fixed in the room (fixed={p.get('fixed')})")
            continue
        m = np.array(p["matrix"], dtype=float).reshape(4, 4, order="F")
        err = angle_deg(-m[:3, 2], gaze)
        if err > 4.0:
            failures.append(f"{key}: image shown {err:.1f} deg away from where the operator looks")
        expect_h = 2.0 * distance * np.tan(np.radians(CAM_VFOV) / 2.0)
        if abs(float(p.get("height", 0.0)) - expect_h) > 0.01:
            failures.append(f"{key}: image {p.get('height')} m tall at {distance} m, life-size is {expect_h:.3f} m")
    line = f"view: {'stereo' if stereo else 'mono'}, {len(keys)} image plane(s), life-size, fixed in the room"
    if stereo and all(planes[k].get("matrix") for k in ("background-left", "background-right")):
        centers = [np.array(planes[k]["matrix"], dtype=float).reshape(4, 4, order="F")[:3, 3]
                   for k in ("background-left", "background-right")]
        sep = float(np.linalg.norm(centers[1] - centers[0]))
        line += f", eye images {sep * 1000:.0f} mm apart"
        if abs(sep - 0.064) > 0.005:
            failures.append(f"left/right eye images {sep * 1000:.0f} mm apart, expected 64 mm")
    info.append(line + f", {angle_deg(-np.array(planes[keys[0]]['matrix']).reshape(4, 4, order='F')[:3, 2], gaze):.1f} deg "
                f"from the operator's gaze")


def press(hs, nodes, name, side="left", hold=0.25):
    node = nodes.get(f"ui-button-{name}")
    if node is None or not node.get("matrix"):
        return None
    T = matrix(node)
    c, n = T[:3, 3], T[:3, 2]
    for k in range(13):
        hs.poke(side, c + n * (0.08 - 0.09 * k / 12))
        time.sleep(1 / 30)
    t0 = time.time()
    time.sleep(hold)
    t1 = time.time()
    for k in range(13):
        hs.poke(side, c + n * (-0.01 + 0.19 * k / 12))
        time.sleep(1 / 30)
    return t0, t1


def wait_for(hs, keys, timeout=10.0):
    t_end = time.time() + timeout
    while time.time() < t_end:
        nodes, _ = hs.scene()
        if all(k in nodes for k in keys):
            return nodes
        time.sleep(0.1)
    return hs.scene()[0]


def main() -> int:
    p = argparse.ArgumentParser(description="End-to-end test of xr_teleoperate's hand path with a simulated headset (fake_headset.py)")
    p.add_argument("--start-delay", type=float, default=2.0)
    p.add_argument("--ramp-time", type=float, default=1.0)
    p.add_argument("--speed-limit", type=float, default=5.0)
    p.add_argument("--hand-mapping", choices=["exact", "relative", "unitree"], default="exact")
    p.add_argument("--view", choices=["lab", "camera"], default="lab")
    p.add_argument("--vr-buttons", type=int, default=1)
    p.add_argument("--record", type=int, default=0, help="1 = the teleop runs with --record: test the REC button")
    p.add_argument("--head-tracking", type=int, default=1)
    p.add_argument("--immersive-view", type=int, default=1)
    p.add_argument("--whole-body", type=int, default=0, help="1 = the teleop runs with --whole-body 1: test walking")
    p.add_argument("--body", choices=["exact", "balance"], default="exact",
                   help="the simulator's robot body (SIM_BODY): exact = placed by pose commands, balance = leg controller")
    p.add_argument("--url", default="wss://127.0.0.1:8012/")
    p.add_argument("--dump", default=None, help="save the recorded topics and the step times to this JSON file")
    a = p.parse_args()
    pose_cmd = POSE_COMMANDS[0] = a.whole_body and a.body == "exact"
    exact, relative = a.hand_mapping == "exact", a.hand_mapping == "relative"
    lab_view = a.view == "lab"

    offsets = palm_offsets()
    rec = Recorder(offsets if a.hand_mapping != "unitree" else {s: UPSTREAM_IK_POINT for s in ("left", "right")})
    ChannelFactoryInitialize(1, "lo")
    subs = []
    for topic, typ, fn in (("rt/lowcmd", LowCmd_, rec.on_lowcmd), ("rt/arm_sdk", LowCmd_, rec.on_lowcmd),
                           ("rt/lowstate", LowState_, rec.on_lowstate), ("rt/brainco/right/state", MotorStates_, rec.on_hand),
                           ("rt/sim_state", String_, rec.on_sim_state), ("rt/sim/body_cmd", String_, rec.on_body_cmd),
                           ("rt/reset_pose/cmd", String_, rec.on_reset)):
        sub = ChannelSubscriber(topic, typ)
        sub.Init(fn, 50)
        subs.append(sub)
    reset_pub = ChannelPublisher("rt/reset_pose/cmd", String_)
    reset_pub.Init()

    start = EXACT_START if exact else START_POSE
    move = EXACT_MOVE if exact else MOVE
    probe_speed = 0.01 if exact else 0.02

    origin = np.zeros(3)
    eye_pt = eye_in_pelvis()
    raised = np.array(start["right"]) + [0.0, 0.06, 0.0]

    def hands(right=None, curl=0.0):
        hs.set_hands(left=tuple(np.array(start["left"]) + origin),
                     right=tuple((np.array(right) if right is not None else np.array(start["right"])) + origin), curl=curl)

    def body_at(offset=(0.0, 0.0, 0.0), yaw=0.0, pitch=TABLE[1], right=None):
        c, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        R = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])
        eye = np.array(HEAD) + origin + offset
        axis = eye - head_matrix(*TABLE, tuple(eye))[:3, :3] @ NECK_TO_EYE
        axis[1] = 0.0
        head = axis + R @ (eye - axis)
        hs.set_head(TABLE[0] + yaw, pitch, position=tuple(head))
        pose = {"left": np.array(start["left"]), "right": np.array(right if right is not None else start["right"])}
        hs.set_hands(**{side: tuple(head + R @ (pose[side] - HEAD)) for side in ("left", "right")})

    def look(yaw, pitch):
        eye = np.array(HEAD) + origin
        neck = eye - head_matrix(*TABLE, tuple(eye))[:3, :3] @ NECK_TO_EYE
        hs.set_head(yaw, pitch, position=tuple(neck + head_matrix(yaw, pitch, tuple(eye))[:3, :3] @ NECK_TO_EYE))

    def walk_to_robot(seconds, attempts=3):
        for attempt in range(attempts):
            nodes = hs.scene()[0]
            if "lab" not in nodes or "link-pelvis_link" not in nodes:
                return
            d = drawn_eye(nodes, eye_pt) - (np.array(HEAD) + origin)
            d[1] = 0.0
            if attempt and np.linalg.norm(d) < 0.10:
                return
            n = max(1, int((seconds if attempt == 0 else 0.5) * 30))
            for k in range(1, n + 1):
                body_at(d * k / n, right=raised)
                time.sleep(1 / 30)
            origin[:] = origin + d
            time.sleep(0.5)

    def scene_at_go(nodes_before, timeout):
        t_end = time.time() + timeout
        while time.time() < t_end:
            nodes = hs.scene()[0]
            if "lab" in nodes and "lab" in nodes_before and nodes["lab"].get("matrix") != nodes_before["lab"].get("matrix"):
                time.sleep(0.03)
                return hs.scene()[0]
            time.sleep(0.01)
        return hs.scene()[0]

    def wait_still(timeout, settle=1.0):
        t_end, t_min = time.time() + timeout, time.time() + settle
        while time.time() < t_end:
            tb_, b_ = rec.get("body")
            recent = window(tb_, b_, time.time() - settle, time.time())
            if time.time() >= t_min and len(recent) and body_still(recent):
                return
            time.sleep(0.1)

    failures, info = [], []
    hs = FakeHeadset(a.url, rate=60.0).start()
    hs.set_head(*TABLE)
    expected = (["lab", "lab-table"] if lab_view else []) + \
               (["ui-button-exit", "ui-button-reset", "ui-button-restart", "ui-button-start", "ui-button-rec"] if a.vr_buttons else [])
    nodes = wait_for(hs, expected + (["link-right_hand_palm_link"] if lab_view else []), timeout=12.0)
    missing = [k for k in expected if k not in nodes]
    if missing:
        failures.append(f"the headset did not receive {missing}")
    if lab_view:
        nodes, parents = hs.scene()
        links = [k for k in nodes if k.startswith("link-")]
        meshes = [k for k in nodes if k.startswith("mesh-")]
        boxes = [k for k in nodes if k.startswith("lab-")]
        info.append(f"lab view: {len(boxes)} room/table boxes, {len(links)} robot links with {len(meshes)} meshes, "
                    f"objects {sorted(k for k in nodes if k.startswith('obj-') and not k.endswith('-shape'))}")
        if len(links) < 50 or len(meshes) < 50 or any(parents.get(k) != "lab" for k in links):
            failures.append(f"lab view: {len(links)} robot links / {len(meshes)} meshes (expected > 50 each, in the lab group)")
        if "link-head_yaw_link" in nodes:
            failures.append("lab view: the robot's head is drawn (it would be around the operator's eyes)")

    hands()
    time.sleep(1.0)
    marks = {}
    if a.vr_buttons:
        touched = press(hs, hs.scene()[0], "start")
        if touched is None:
            print("[FAIL] no START button in the headset")
            hs.stop()
            return 1
        marks["r"] = touched[0]
    else:
        marks["r"] = time.time()
        if ipc("CMD_START") is None:
            print("[FAIL] no IPC reply to CMD_START (is teleop_hand_and_arm.py running with --ipc?)")
            hs.stop()
            return 1
    hs.hide_hands()
    time.sleep(a.start_delay + 1.0)
    marks["controllers"] = time.time()
    hands()
    hs.use_controllers(True)
    time.sleep(a.start_delay + 1.0)
    marks["hands"] = time.time()
    hs.use_controllers(False)
    right = np.array(start["right"])
    t0 = time.time()
    board_seen = False
    while time.time() - t0 < a.start_delay + a.ramp_time + 1.2:
        hands(right + [0.0, probe_speed * (time.time() - t0), 0.0])
        if time.time() - t0 > 0.8 and time.time() - t0 < a.start_delay:
            board_seen |= "ui-board" in hs.scene()[0]
        time.sleep(1 / 30)
    right = right + [0.0, probe_speed * (time.time() - t0), 0.0]
    hands(right)
    time.sleep(1.5)
    marks["move"] = time.time()
    still_nodes = hs.scene()[0]
    palm_before = operator_palm_centre(hs.joints("right"))
    for k in range(1, 31):
        hands(right + move * k / 30)
        time.sleep(1 / 30)
    right = right + move
    time.sleep(1.5)
    marks["moved"] = time.time()
    moved_nodes = hs.scene()[0]
    palm_after = operator_palm_centre(hs.joints("right"))
    for k in range(1, 31):
        look(*(np.array(TABLE) + (np.array(LOOK) - TABLE) * k / 30))
        time.sleep(1 / 30)
    time.sleep(2.0)
    marks["looked"] = time.time()
    planes = hs.image_planes()
    for k in range(29, -1, -1):
        look(*(np.array(TABLE) + (np.array(LOOK) - TABLE) * k / 30))
        time.sleep(1 / 30)
    time.sleep(1.5)
    marks["curl"] = time.time()
    hands(right, curl=1.0)
    time.sleep(1.5)
    marks["curled"] = time.time()
    hands(right, curl=0.0)
    time.sleep(1.0)
    marks["glitch"] = time.time()
    hs.glitch("right", (0.0, 0.0, 0.3))
    time.sleep(1.5)
    marks["lost"] = time.time()
    hs.hide_hands()
    time.sleep(1.5)
    marks["back"] = time.time()
    hands()
    time.sleep(a.ramp_time + 1.5)
    marks["again"] = time.time()
    for k in range(1, 31):
        hands(np.array(start["right"]) + [0.0, 0.06 * k / 30, 0.0])
        time.sleep(1 / 30)
    time.sleep(1.5)
    marks["raised"] = time.time()
    centre = np.array(start["right"]) + [0.0, 0.05, -0.05]
    here = np.array(start["right"]) + [0.0, 0.06, 0.0]
    traces = []
    marks["trace"] = time.time()
    for radius, period in TRACES:
        ring = lambda ang: centre + radius * np.array([np.cos(ang), np.sin(ang), 0.0])
        for k in range(1, 16):
            hands(here + (ring(0.0) - here) * k / 15)
            time.sleep(1 / 30)
        time.sleep(1.0)
        trace, n = [], int(period * 30)
        t_ring = time.time()
        for k in range(1, 2 * n + 1):
            hands(ring(2 * np.pi * (time.time() - t_ring) / period))
            time.sleep(1 / 30)
            if lab_view and k > n / 4:
                try:
                    trace.append(float(np.linalg.norm(drawn_palm_centre(hs.scene()[0], "right") - operator_palm_centre(hs.joints("right")))))
                except KeyError:
                    pass
        traces.append((2 * np.pi * radius / period, trace))
        for k in range(1, 16):
            hands(ring(0.0) + (here - ring(0.0)) * k / 15)
            time.sleep(1 / 30)
    marks["traced"] = time.time()
    time.sleep(1.5)
    step_drawn = []
    if lab_view:
        marks["step"] = time.time()
        while time.time() < marks["step"] + STEP_TIME:
            u = (time.time() - marks["step"]) / STEP_TIME
            hands(here + [0.0, STEP_SIZE * (10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5), 0.0])
            try:
                nodes = hs.scene()[0]
                step_drawn.append((nodes["link-right_hand_palm_link"].get("received", time.time()), drawn_palm_centre(nodes, "right")))
            except KeyError:
                pass
            time.sleep(0.004)
        hands(here + [0.0, STEP_SIZE, 0.0])
        while time.time() < marks["step"] + 0.8:
            try:
                nodes = hs.scene()[0]
                step_drawn.append((nodes["link-right_hand_palm_link"].get("received", time.time()), drawn_palm_centre(nodes, "right")))
            except KeyError:
                pass
            time.sleep(0.002)
        hands(here)
        marks["stepped"] = time.time()
        marks["step_sent"] = hs.sent_at(marks["step"])
        time.sleep(1.0)
    marks["walk"] = time.time()
    walk_nodes = {}
    if a.whole_body:
        walk_nodes["before"] = hs.scene()[0]
        for k in range(1, 76):
            body_at((0.0, 0.0, 0.6 * k / 75))
            time.sleep(1 / 30)
        marks["walked"] = time.time()
        wait_still(10.0)
        marks["arrived"] = time.time()
        walk_nodes["after"] = hs.scene()[0]
        walk_nodes["palm"] = operator_palm_centre(hs.joints("right"))
        for k in range(74, -1, -1):
            body_at((0.0, 0.0, 0.6 * k / 75))
            time.sleep(1 / 30)
        wait_still(10.0)
        marks["squat"] = time.time()
        for k in range(1, 31):
            body_at((0.0, -0.15 * k / 30, 0.0))
            time.sleep(1 / 30)
        time.sleep(2.0)
        marks["squatted"] = time.time()
        for k in range(29, -1, -1):
            body_at((0.0, -0.15 * k / 30, 0.0))
            time.sleep(1 / 30)
        time.sleep(1.5)
        marks["turn"] = time.time()
        for k in range(1, 46):
            body_at(yaw=70.0 * k / 45)
            time.sleep(1 / 30)
        wait_still(10.0)
        marks["turned"] = time.time()
        for k in range(44, -1, -1):
            body_at(yaw=70.0 * k / 45)
            time.sleep(1 / 30)
        wait_still(10.0)
        hands()
        hs.set_head(*TABLE)
    marks["walk_done"] = time.time()
    rec_touch = []
    if a.record and a.vr_buttons:
        for _ in range(2):
            touched = press(hs, hs.scene()[0], "rec")
            if touched is None:
                failures.append("no REC button in the headset")
                break
            rec_touch.append(touched)
            hands(np.array(start["right"]) + [0.0, 0.06, 0.0])
            time.sleep(2.5)
    restart, reset, simreset, ended = {}, {}, {}, {}
    if a.vr_buttons:
        hands(raised)
        time.sleep(1.5)
        restart["ref"] = time.time()
        for k in range(1, 31):
            body_at((0.10 * k / 30, 0.0, 0.0), right=raised)
            time.sleep(1 / 30)
        origin[:] = [0.10, 0.0, 0.0]
        restart["head"] = np.array(HEAD) + origin
        time.sleep(1.5)
        restart["shifted"] = time.time()
        restart["nodes_before"] = hs.scene()[0]
        touched = press(hs, hs.scene()[0], "restart")
        if touched is None:
            failures.append("no RESTART button in the headset")
        else:
            restart["touch"] = touched
            hands(raised)
            restart["nodes_go"] = scene_at_go(restart["nodes_before"], a.start_delay + 4.0)
            time.sleep(max(0.0, touched[1] + a.start_delay + a.ramp_time + 2.0 - time.time()))
            restart["done"] = time.time()
            restart["nodes_after"] = hs.scene()[0]
    if a.vr_buttons:
        reset["first"] = time.time()
        if press(hs, hs.scene()[0], "reset") is None:
            failures.append("no RESET button in the headset")
            reset.clear()
        else:
            time.sleep(1.5)
            reset["second"] = time.time()
            nodes_before = hs.scene()[0]
            press(hs, hs.scene()[0], "reset")
            hands(raised)
            time.sleep(1.0)
            if a.whole_body:
                walk_to_robot(1.0)
            reset["head"] = np.array(HEAD) + origin
            reset["nodes_go"] = scene_at_go(nodes_before, a.start_delay + 6.0)
            time.sleep(max(0.0, reset["second"] + 1.0 + 0.5 + a.start_delay + a.ramp_time + 2.0 - time.time()))
            reset["done"] = time.time()
            reset["nodes_after"] = hs.scene()[0]
    if a.whole_body:
        for k in range(1, 61):
            body_at((0.0, 0.0, 0.5 * k / 60), right=raised)
            time.sleep(1 / 30)
        origin[:] = origin + [0.0, 0.0, 0.5]
        time.sleep(3.0)
        simreset["sent"] = time.time()
        reset_pub.Write(String_(data="2"))
        time.sleep(2.5)
        simreset["away"] = time.time()
        walk_to_robot(2.0)
        simreset["back"] = time.time()
        time.sleep(a.start_delay + a.ramp_time + 2.0)
        simreset["done"] = time.time()
    marks["q"] = time.time()
    if a.vr_buttons:
        ended["first"] = time.time()
        if press(hs, hs.scene()[0], "exit") is None:
            failures.append("no EXIT button in the headset")
            ended.clear()
            ipc("CMD_STOP")
        else:
            time.sleep(1.5)
            ended["second"] = marks["q"] = time.time()
            press(hs, hs.scene()[0], "exit")
            t_end = time.time() + 3.0
            while time.time() < t_end and "ui-ended" not in hs.scene()[0]:
                time.sleep(0.05)
            ended["panel"] = "ui-ended" in hs.scene()[0]
    else:
        ipc("CMD_STOP")
    t_end = time.time() + 15.0
    while time.time() < t_end:
        tc, _ = rec.get("cmd")
        if len(tc) and time.time() - tc[-1] > 1.0:
            break
        time.sleep(0.2)
    ended["stopped"] = time.time() - (rec.get("cmd")[0][-1] if len(rec.get("cmd")[0]) else 0.0) > 1.0
    hs.stop()

    if a.dump:
        with rec.lock, open(a.dump, "w") as f:
            json.dump({"data": rec.data, "marks": marks, "step_drawn": [(t_, p_.tolist()) for t_, p_ in step_drawn], "restart": {k: v for k, v in restart.items() if "nodes" not in k},
                       "reset": {k: v for k, v in reset.items() if "nodes" not in k}, "simreset": simreset,
                       "ended": ended}, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    t, q = rec.get("cmd")
    if len(t) == 0:
        print("[FAIL] no rt/lowcmd from xr_teleoperate")
        return 1
    th_, hq = rec.get("head")
    q_ref = q[t < marks["r"]][-1]
    dev = np.max(np.abs(q - q_ref), axis=1)

    early = window(t, dev, marks["r"] + 0.1, marks["controllers"])
    if len(early) and early.max() > 1e-6:
        failures.append(f"the arms moved ({early.max():.3f} rad) before any hand was tracked")
    ctrl = window(t, dev, marks["controllers"], marks["hands"])
    if len(ctrl) and ctrl.max() > 1e-6:
        failures.append(f"the arms moved ({ctrl.max():.3f} rad) while a controller was in use")
    start_idx = np.flatnonzero((dev > 1e-3) & (t > marks["hands"]))
    if not len(start_idx):
        print("[FAIL] the arms never moved after the hands appeared" + (" (START pressed in VR)" if a.vr_buttons else ""))
        return 1
    t_go = t[start_idx[0]]
    hold = t_go - marks["hands"]
    if hold < a.start_delay - 0.15:
        failures.append(f"the arms moved {hold:.2f} s after the controllers were put down, before the {a.start_delay:g} s countdown ended")
    if hold > a.start_delay + 1.6:
        failures.append(f"the arms started only {hold:.2f} s after the controllers were put down")
    line = (("START pressed in VR; " if a.vr_buttons else "") + "start: arms held with no hands and while a controller "
            f"moved; {hold:.2f} s after the controllers were put down they followed")
    if not relative and a.ramp_time > 0:
        quarter = a.ramp_time / 4
        frac = window(t, dev, t_go, t_go + quarter).max() / max(window(t, dev, t_go, marks["move"]).max(), 1e-9)
        line += f", {frac:.0%} of the start-up motion in the first {quarter:.2f} s"
        if frac > 0.5:
            failures.append(f"{frac:.0%} of the start-up motion done {quarter:.2f} s after the start: no blend-in")
    info.append(line)
    if lab_view and not board_seen:
        failures.append("lab view: no status board during the countdown")

    te, ee = rec.get("ee")
    world_scale = float(np.linalg.norm(matrix(still_nodes["lab"])[:3, 0])) if lab_view and "lab" in still_nodes else 1.0
    info.append(f"world scale {world_scale:.2f}")
    expect = np.array([-move[2], -move[0], move[1]])
    if exact:
        expect = expect / world_scale
    if len(te):
        d = mean_change(te, ee, marks["move"], marks["moved"])
        gain = float(d @ expect / (expect @ expect))
        info.append(f"follow: hand moved {np.round(expect * 100, 1)} cm, robot hand moved {np.round(d * 100, 1)} cm (gain {gain:.2f})")
        if gain < (0.8 if a.hand_mapping != "unitree" else 0.5):
            failures.append(f"the robot's hand did not follow the operator's hand (gain {gain:.2f})")
    else:
        change = float(np.max(np.abs(mean_change(t, q, marks["move"], marks["moved"]))))
        info.append(f"follow: arm targets changed by {change:.2f} rad when the hand moved (mock simulator, no rt/sim_state)")
        if change < 0.05:
            failures.append("the arm targets did not change when the hand moved")
    sim = "isaac" if len(te) else "mock"
    warnings = []

    def physical(msg):
        (warnings if a.whole_body and sim == "isaac" and a.body == "balance" else failures).append(msg)
    if exact and lab_view:
        errs = []
        for label, scene_nodes, palm in (("still", still_nodes, palm_before), ("moved", moved_nodes, palm_after)):
            try:
                errs.append(float(np.linalg.norm(drawn_palm_centre(scene_nodes, "right") - palm)))
            except KeyError as e:
                failures.append(f"co-location: the headset has no {e}")
        if errs:
            info.append(f"co-location: the robot's right palm as drawn in the headset is {errs[0] * 100:.1f} cm (hand still) and "
                        f"{errs[-1] * 100:.1f} cm (after the move) from the operator's palm")
            if max(errs) > COLOCATION_TOLERANCE[sim]:
                physical(f"exact mapping: the drawn robot palm is {max(errs) * 100:.1f} cm from the operator's palm "
                                f"(tolerance {COLOCATION_TOLERANCE[sim] * 100:.1f} cm)")
        for speed, trace in traces:
            if trace:
                err = np.array(trace)
                info.append(f"working: while the right hand drew circles at {speed * 100:.0f} cm/s the drawn robot palm was {err.mean() * 100:.1f} "
                            f"cm (mean) / {err.max() * 100:.1f} cm (max) from the operator's palm")
                if speed < 0.2 and err.mean() > TRACE_TOLERANCE[sim]:
                    physical(f"exact mapping: while the hand moved the drawn robot palm was {err.mean() * 100:.1f} cm (mean) from the "
                             f"operator's palm (tolerance {TRACE_TOLERANCE[sim] * 100:.1f} cm)")
        if step_drawn:
            def half(ts, values, t0):
                values = np.asarray(values, dtype=float)
                if not len(values) or not np.any(ts > t0 + 0.5):
                    return np.nan
                before = values[np.argmin(np.abs(ts - t0))]
                final = values[ts > t0 + 0.5].mean(axis=0) - before
                if np.linalg.norm(final) < 1e-4:
                    return np.nan
                progress = (values - before) @ final / (final @ final)
                after = np.flatnonzero((ts >= t0) & (progress >= 0.5))
                return float(ts[after[0]] - t0) if len(after) else np.nan
            t0 = marks["step_sent"]
            td = np.array([s_[0] for s_ in step_drawn])
            win = lambda key: [(np.array(tt), vv) for tt, vv in [rec.get(key)]][0]
            tc_, qc_ = win("cmd")
            tq_, qq_ = win("q")
            sel_c = (tc_ > t0 - 0.3) & (tc_ < t0 + 0.8)
            sel_q = (tq_ > t0 - 0.3) & (tq_ < t0 + 0.8)
            lat = {"command": half(tc_[sel_c], qc_[sel_c], t0) if np.any(sel_c) else np.nan,
                   "arm": half(tq_[sel_q], qq_[sel_q], t0) if np.any(sel_q) else np.nan,
                   "drawn": half(td, [s_[1] for s_ in step_drawn], t0)}
            lat = {k: v - STEP_TIME / 2 for k, v in lat.items()}
            pd = np.array([s_[1] for s_ in step_drawn])
            rise = pd[:, 1] - pd[0, 1]
            final = rise[td > t0 + 0.5].mean() if np.any(td > t0 + 0.5) else np.nan
            info.append(f"latency: the hand moved {STEP_SIZE * 100:.0f} cm in {STEP_TIME:.2f} s and stopped - half way after the hand: " + ", ".join(
                f"{k} {v * 1000:.0f} ms" for k, v in lat.items()) + f" (from when the headset sent it); the drawn palm overshot "
                f"by {max(0.0, rise.max() - final) * 100:.1f} cm, {len(np.unique(td)) / 0.8:.0f} updates/s")
        tw, wp = rec.get("wrist_pose")
        if len(tw) and "link-right_wrist_roll_link" in moved_nodes:
            node = moved_nodes["link-right_wrist_roll_link"]
            if a.whole_body:
                sim_wrist = wp[int(np.argmin(np.abs(tw - node["received"])))].reshape(4, 4)[:3, 3]
            else:
                sim_wrist = window(tw, wp, marks["moved"] - 0.5, marks["moved"]).mean(axis=0).reshape(4, 4)[:3, 3]
            off = float(np.linalg.norm(sim_wrist - matrix(node)[:3, 3]))
            info.append(f"lab view: the drawn right wrist is {off * 1000:.1f} mm from the simulator's (rt/sim_state)")
            if off > (0.015 if a.whole_body else 0.005):
                physical(f"lab view: the drawn robot is {off * 1000:.1f} mm away from the simulated one")

    if a.head_tracking:
        head_q = window(th_, hq, marks["looked"] - 0.5, marks["looked"]).mean(axis=0)
        err = angle_deg(camera_forward(head_q), gaze_robot(*LOOK))
        line = (f"look {LOOK[0]:g} deg left, {LOOK[1]:g} deg down: head pitch {np.degrees(head_q[0]):.1f}, yaw "
                f"{np.degrees(head_q[1]):.1f} deg, camera {err:.1f} deg from the gaze")
        if err > 3.0:
            failures.append(f"the robot's head camera points {err:.1f} deg away from where the operator looks")
        tl, hl = rec.get("head_link")
        if len(tl):
            link = window(tl, hl, marks["looked"] - 0.5, marks["looked"]).mean(axis=0)
            off = angle_deg(link, head_link_rotation(head_q)[:, 0])
            line += f"; simulated head {off:.1f} deg from the commanded pose"
            if off > 4.0:
                failures.append(f"the simulated head is {off:.1f} deg away from the commanded pose")
        info.append(line)
        back_q = window(th_, hq, marks["curl"] - 0.3, marks["curl"]).mean(axis=0)
        back_err = angle_deg(camera_forward(back_q), gaze_robot(*TABLE))
        if back_err > 3.0:
            failures.append(f"the head did not turn back to the table with the operator ({back_err:.1f} deg off)")
    if a.immersive_view and not lab_view:
        check_view(planes, failures, info)
    if a.hand_mapping != "unitree" and a.whole_body and len(te):
        p0 = window(te, ee, marks["moved"] - 0.3, marks["moved"]).mean(axis=0)
        drift = float(np.max(np.linalg.norm(window(te, ee, marks["moved"], marks["curl"]) - p0, axis=1)))
        info.append(f"the robot's right palm while the operator looked around: moved {drift * 100:.1f} cm")
        if drift > 0.025:
            physical(f"the robot's palm moved {drift * 100:.1f} cm when only the operator's head moved")
    elif a.hand_mapping != "unitree":
        moved = float(np.max(np.abs(window(t, q, marks["moved"], marks["curl"]) - window(t, q, marks["moved"] - 0.3, marks["moved"]).mean(axis=0))))
        info.append(f"arms while the operator looked around: moved {moved:.3f} rad")
        if moved > 0.02:
            failures.append(f"the arms moved {moved:.3f} rad when only the operator's head moved")

    th, h = rec.get("hand")
    if len(th):
        open_ = window(th, h, marks["curl"] - 0.5, marks["curl"]).mean(axis=0)
        closed = window(th, h, marks["curled"] - 0.5, marks["curled"]).mean(axis=0)
        rise = float(np.mean(closed[2:] - open_[2:]))
        info.append(f"fingers: robot's right hand (0 open .. 1 closed) {np.round(open_, 2)} -> {np.round(closed, 2)}")
        if rise < 0.2:
            failures.append(f"closing the hand closed the robot's fingers by only {rise:.2f}")
    else:
        failures.append("no rt/brainco/right/state from the simulator")

    before = window(t, q, marks["glitch"] - 0.3, marks["glitch"]).mean(axis=0)
    if a.whole_body and sim == "isaac":
        glitch_end, before = marks["glitch"] + 0.4, q[t <= marks["glitch"]][-1]
    else:
        glitch_end = marks["lost"]
    jump = float(np.max(np.abs(window(t, q, marks["glitch"], glitch_end) - before)))
    info.append(f"tracking glitch (one frame, wrist 30 cm off): arms moved {jump:.3f} rad")
    if jump > 0.02:
        physical(f"a one-frame tracking glitch moved the arms {jump:.3f} rad")

    held = window(t, q, marks["lost"] + 0.5, marks["back"])
    drift = float(np.max(np.abs(held - held[0]))) if len(held) else 0.0
    if drift > 1e-6:
        failures.append(f"the arms moved {drift:.3f} rad while no hand was tracked")
    back = float(np.max(np.abs(window(t, q, marks["again"] - 0.3, marks["again"]).mean(axis=0) - held[-1]))) if len(held) else 0.0
    end_again = marks["raised"]
    again = float(np.max(np.abs(mean_change(t, q, marks["again"], end_again))))
    if relative:
        if back > 0.05:
            failures.append(f"relative mode: the arms moved {back:.2f} rad when the hands reappeared elsewhere (should re-anchor)")
        if again < 0.02:
            failures.append("relative mode: the arms did not follow again after the hands reappeared")
        info.append(f"hands lost: arms held (drift {drift:.4f} rad); back elsewhere: arms stayed ({back:.3f} rad), "
                    f"then followed again ({again:.2f} rad)")
    else:
        if back < 0.02:
            failures.append("the arms did not follow again after the hands came back")
        info.append(f"hands lost: arms held (drift {drift:.4f} rad); hands back: arms moved {back:.2f} rad")

    if a.whole_body:
        tb, body = rec.get("body")
        tr, root = rec.get("root")
        if not len(tb) or not len(tr):
            failures.append(f"whole body: no {'rt/sim/body_cmd' if not len(tb) else 'rt/sim_state robot pose'}")
        else:
            w = window(tb, body, marks["trace"], marks["traced"])
            if len(w):
                dh, dp = float(np.ptp(w[:, 3])), float(np.ptp(w[:, 4]))
                info.append(f"whole body while a hand worked within reach: body height {w[:, 3].min():.3f}-{w[:, 3].max():.3f} m, "
                            f"lean {w[:, 4].min():.2f}-{w[:, 4].max():.2f} rad")
                if dh > 0.01 or dp > 0.03:
                    physical(f"whole body: the robot bent ({dh * 100:.1f} cm, {dp:.2f} rad) while the operator stood and moved a "
                             f"hand within its reach")
            r0 = window(tr, root, marks["walk"] - 0.5, marks["walk"]).mean(axis=0)
            r1 = window(tr, root, marks["arrived"] - 0.5, marks["arrived"]).mean(axis=0)
            fwd = np.array([np.cos(r0[3]), np.sin(r0[3])])
            walked = -float((r1[:2] - r0[:2]) @ fwd)
            v_walk = -float(window(tb, body, marks["walk"] + 0.5, marks["walked"])[:, 0].mean()) if not pose_cmd else \
                float(np.linalg.norm(np.diff(window(tb, body, marks["walk"], marks["walked"])[[0, -1], :2], axis=0)) / max(marks["walked"] - marks["walk"], 1e-3))
            v_stop = body_motion(window(tb, body, marks["arrived"] - 0.5, marks["arrived"]))
            line = (f"whole body: operator walked 0.60 m backward: body command {v_walk:.2f} m/s backward, robot walked "
                    f"{walked:.2f} m and stopped (command {v_stop:.2f} at the end)")
            if walked * world_scale < 0.45 or walked * world_scale > 0.70:
                physical(f"whole body: the robot walked {walked:.2f} m when the operator walked 0.60 m")
            if v_stop > 1e-6:
                physical(f"whole body: the robot kept walking ({v_stop:.2f}) after the operator stopped")
            if "link-pelvis_link" in walk_nodes.get("after", {}) and "link-pelvis_link" in walk_nodes.get("before", {}):
                drawn = matrix(walk_nodes["after"]["link-pelvis_link"])[:3, 3] - matrix(walk_nodes["before"]["link-pelvis_link"])[:3, 3]
                off = abs(-float(drawn[:2] @ fwd) - walked)
                line += f"; the headset drew it {-float(drawn[:2] @ fwd):.2f} m further back"
                if off > 0.03:
                    physical(f"whole body: the headset drew the robot {off * 100:.1f} cm away from where it walked")
                if exact and lab_view:
                    err = float(np.linalg.norm(drawn_palm_centre(walk_nodes["after"], "right") - walk_nodes["palm"]))
                    line += f"; its right palm then {err * 100:.1f} cm from the operator's"
                    if err > COLOCATION_TOLERANCE[sim]:
                        physical(f"whole body: after walking the drawn robot palm is {err * 100:.1f} cm from the operator's palm")
            h0 = float(window(tr, root, marks["squat"] - 0.5, marks["squat"])[:, 2].mean())
            h1 = float(window(tr, root, marks["squat"], marks["squatted"])[:, 2].min())
            h_min = loco_spec.H_MIN if a.body == "exact" else loco_spec.USE_H_MIN
            line += f"; squat 15 cm: pelvis {h0:.3f} -> {h1:.3f} m at the lowest (lowest allowed {h_min:.2f} m)"
            if h0 - h1 < 0.03 and h1 > h_min + 0.025:
                physical(f"whole body: the operator squatted 15 cm, the robot's pelvis went down {(h0 - h1) * 100:.1f} cm")
            y0 = float(window(tr, root, marks["turn"] - 0.5, marks["turn"])[:, 3].mean())
            y1 = float(window(tr, root, marks["turned"] - 0.5, marks["turned"])[:, 3].mean())
            turned = float(np.degrees((y1 - y0 + np.pi) % (2 * np.pi) - np.pi))
            line += f"; turned 70 deg: robot turned {turned:.0f} deg"
            if turned < 25.0:
                physical(f"whole body: the operator turned 70 deg, the robot {turned:.0f} deg")
            looked = window(tr, root, marks["moved"] - 0.5, marks["curl"])
            if len(looked):
                yaw_look = float(np.degrees(np.max(np.abs((looked[:, 3] - looked[0, 3] + np.pi) % (2 * np.pi) - np.pi))))
                line += f"; looking 30 deg aside: the robot turned {yaw_look:.1f} deg"
                if yaw_look > 15.0:
                    physical(f"whole body: looking 30 deg aside turned the robot {yaw_look:.1f} deg")
            info.append(line)

    if rec_touch:
        at_button = max(float(np.max(np.abs(window(t, q, t0_, t1_) - window(t, q, t0_, t1_)[0]))) for t0_, t1_ in rec_touch)
        info.append(f"REC pressed twice in VR; arms while the fingertip was at the button: moved {at_button:.4f} rad")
        if at_button > 1e-6:
            failures.append(f"the arms moved {at_button:.4f} rad while a fingertip was at a VR button")

    eye = eye_in_pelvis()
    tol = COLOCATION_TOLERANCE[sim]
    start_tol = 0.10 if sim == "isaac" else 0.02
    tb, body = rec.get("body")
    tr, root = rec.get("root")
    if restart.get("touch"):
        t0_, t1_ = restart["touch"]
        held = window(t, q, t0_, t1_ + a.start_delay)
        drift = float(np.max(np.abs(held - held[0]))) if len(held) else 0.0
        line = f"RESTART pressed in VR: arms held (drift {drift:.4f} rad)"
        if drift > 1e-6:
            failures.append(f"RESTART: the arms moved {drift:.4f} rad before GO")
        if exact and len(held):
            moved = np.flatnonzero((t > t1_) & (np.max(np.abs(q - held[-1]), axis=1) > 1e-3))
            if not len(moved):
                failures.append("RESTART: the arms did not follow again after GO")
            else:
                wait = float(t[moved[0]] - t1_)
                line += f", followed again {wait:.2f} s after the fingertip left the button"
                if wait < a.start_delay - 0.15:
                    failures.append(f"RESTART: the arms followed {wait:.2f} s after the press, before the {a.start_delay:g} s countdown")
                if a.whole_body:
                    stood = window(tb, body, t1_, float(t[moved[0]]) - 0.1)
                    if not len(stood) or not body_still(stood) or np.abs(stood[:, 3] - loco_spec.H_NOM).max() > 1e-6 \
                            or np.abs(stood[:, 4]).max() > 1e-6:
                        failures.append("RESTART: the body command until GO was not 'stand still, upright'")
        if exact and lab_view:
            try:
                d0, dg, d1 = (drawn_eye(restart[k], eye) - restart["head"] for k in ("nodes_before", "nodes_go", "nodes_after"))
                line += (f"; the robot's eyes as drawn were {d0[0] * 100:+.1f} cm sideways from the operator's after they "
                         f"stepped 10 cm aside, {dg[0] * 100:+.1f} cm sideways ({np.linalg.norm(dg) * 100:.1f} cm in all) "
                         f"at GO and {d1[0] * 100:+.1f} cm sideways ({np.linalg.norm(d1) * 100:.1f} cm in all) 2 s later")
                if abs(d0[0]) < 0.05 or abs(dg[0]) > tol or (not a.whole_body and np.linalg.norm(dg) > tol):
                    failures.append(f"RESTART did not put the robot's eyes at the operator's ({d0[0] * 100:+.1f} cm sideways "
                                    f"before, {dg[0] * 100:+.1f} cm at GO)")
            except KeyError as e:
                failures.append(f"RESTART: the headset has no {e}")
        info.append(line)

    if reset.get("second"):
        tq, rv = rec.get("reset")
        f0, s0 = reset["first"], reset["second"]
        early_ = [v for tt, v in zip(tq, rv) if f0 <= tt < s0]
        late = [(tt, v[0]) for tt, v in zip(tq, rv) if s0 <= tt < reset["done"]]
        if early_:
            failures.append("RESET acted on the first press (it needs a second one)")
        if len(late) != 1 or late[0][1] != 2.0:
            failures.append(f"RESET pressed twice: reset requests {[v for _, v in late]} (expected one '2')")
        else:
            t_req = late[0][0]
            at_reset = window(t, q, t_req - 0.01, t_req + 0.3)
            home_err = float(np.max(np.abs(at_reset))) if len(at_reset) else np.inf
            line = (f"RESET pressed twice in VR: one reset request ('2', none after the first press); arms "
                    f"{home_err:.4f} rad from home when it was sent")
            if home_err > 2e-3:
                failures.append(f"RESET: the arms were {home_err:.3f} rad from home when the scene was reset")
            after_q = window(t, q, reset["done"] - 0.4, reset["done"])
            if exact and (not len(after_q) or float(np.max(np.abs(after_q.mean(axis=0)))) < 0.05):
                failures.append("RESET: the arms did not follow the operator again after GO")
            if a.whole_body and len(tr):
                back_at = window(tr, root, t_req + 0.5, t_req + 1.0)
                off = float(np.linalg.norm(back_at[:, :2].mean(axis=0))) if len(back_at) else np.inf
                line += f"; the robot back {off * 100:.1f} cm from its start"
                if off > 0.25:
                    failures.append(f"RESET: the robot is {off * 100:.1f} cm from its start after the reset")
                elif off > start_tol:
                    physical(f"RESET: the robot settled {off * 100:.1f} cm from its start after the reset")
            if exact and lab_view and "lab" in reset["nodes_go"]:
                dg = drawn_eye(reset["nodes_go"], eye) - reset["head"]
                line += f"; at GO its eyes {dg[0] * 100:+.1f} cm sideways ({np.linalg.norm(dg) * 100:.1f} cm in all) from the operator's"
                if abs(dg[0]) > tol or (not a.whole_body and np.linalg.norm(dg) > tol):
                    failures.append(f"RESET: at GO the robot's eyes are {np.linalg.norm(dg) * 100:.1f} cm from the operator's")
            info.append(line)

    if simreset and len(tr):
        ts = simreset["sent"]
        before = window(tr, root, ts - 0.5, ts)
        away = window(tr, root, ts + 0.5, simreset["back"])
        followed = float(np.linalg.norm(before[:, :2].mean(axis=0))) if len(before) else 0.0
        stray = float(np.linalg.norm(away[:, :2], axis=1).max()) if len(away) else np.inf
        arms = window(t, q, ts + 0.3, simreset["away"])
        arm_drift = float(np.max(np.abs(arms - arms[0]))) if len(arms) else np.inf
        legs = window(tb, body, ts + 0.3, simreset["away"])
        walk_cmd = body_motion(legs) if len(legs) else np.inf
        line = (f"robot reset by the simulator 0.5 m from the operator: it had followed {followed:.2f} m, then stayed at its "
                f"start ({stray * 100:.1f} cm) with its arms held ({arm_drift:.4f} rad) and no walking command ({walk_cmd:.2f})")
        if followed * world_scale < 0.3:
            failures.append(f"robot reset: the robot followed the operator only {followed:.2f} m before")
        if stray > 0.25 or arm_drift > 1e-6 or walk_cmd > 1e-6:
            failures.append(f"robot reset: while the operator was away the robot moved {stray * 100:.1f} cm, its arms "
                            f"{arm_drift:.4f} rad, walking command {walk_cmd:.2f}")
        elif stray > start_tol:
            physical(f"robot reset: while the operator was away the robot settled {stray * 100:.1f} cm from its start")
        if exact:
            moved = np.flatnonzero((t > simreset["away"]) & (np.max(np.abs(q - (arms[-1] if len(arms) else 0.0)), axis=1) > 1e-3))
            if not len(moved):
                failures.append("robot reset: the arms did not follow again once the operator was back")
            else:
                line += f"; operator back: the arms followed again after {float(t[moved[0]] - simreset['back']):+.2f} s"
        info.append(line)

    if ended.get("first"):
        alive = window(t, q, ended["first"] + 1.2, ended["second"])
        if not len(alive):
            failures.append("EXIT acted on the first press (it needs a second one)")
        if not ended.get("panel"):
            failures.append("EXIT: the headset was not told that the teleoperation ended (no ui-ended panel)")
        info.append(f"EXIT pressed twice in VR: still running after the first press, 'ended' panel in the headset "
                    f"{'shown' if ended.get('panel') else 'MISSING'}")
    if not ended.get("stopped"):
        failures.append("the teleop kept sending arm commands after EXIT / [q]")

    cut = simreset.get("sent", np.inf)
    speed, speed_t = max(max_speed_at(t[(t > marks["r"]) & (t < cut - 0.05)], q[(t > marks["r"]) & (t < cut - 0.05)]),
                         max_speed_at(t[t > cut + 0.5], q[t > cut + 0.5]) if np.isfinite(cut) else (0.0, None),
                         key=lambda b: b[0])
    if a.speed_limit > 0 and speed > SPEED_TOLERANCE * a.speed_limit:
        after = max(((k, v) for k, v in marks.items() if v <= speed_t), key=lambda kv: kv[1], default=("start", 0.0))
        failures.append(f"arm target speed {speed:.1f} rad/s > limit {a.speed_limit:g} rad/s ({speed_t - after[1]:.2f} s after "
                        f"mark '{after[0]}'; at {speed_t:.2f}: RESTART {restart.get('touch')}, RESET {reset.get('second')}, "
                        f"robot reset {simreset.get('sent')}, EXIT {ended.get('second')})")
    head_speed = max_mean_speed(th_[th_ > marks["r"]], hq[th_ > marks["r"]])
    if head_speed > SPEED_TOLERANCE * HEAD_SPEED_LIMIT:
        failures.append(f"head target speed {head_speed:.1f} rad/s > limit {HEAD_SPEED_LIMIT:g} rad/s")
    td, dq = rec.get("dq")
    peak = float(window(td, dq, marks["r"], marks["q"]).max()) if len(td) else 0.0
    home = float(np.max(np.abs(q[-1])))
    head_home = float(np.max(np.abs(hq[-1])))
    if head_home > 0.02 or home > 0.01:
        failures.append(f"the head is {head_home:.2f} rad and the arms {home:.2f} rad from home after EXIT / [q]")
    info.append(f"max arm target speed {speed:.1f} rad/s (limit {a.speed_limit:g}), head {head_speed:.1f} rad/s "
                f"(limit {HEAD_SPEED_LIMIT:g}); robot's peak arm joint speed {peak:.2f} rad/s; arms {home:.3f} rad and "
                f"head {head_home:.3f} rad from home after [q]")

    for line in info:
        print(f"       {line}")
    for w in warnings:
        print(f"[WARN] teleop hand path (free-standing robot in Isaac Sim): {w}")
    for f in failures:
        print(f"[FAIL] teleop hand path: {f}")
    if not failures:
        checked = (["lab view"] if lab_view else []) + (["START/REC/RESTART/RESET/EXIT in VR"] if a.vr_buttons else []) + \
                  (["robot palm at the operator's palm"] if exact and lab_view else []) + \
                  (["head follows"] if a.head_tracking else []) + (["image fixed in the room"] if a.immersive_view and not lab_view else []) + \
                  (["walks, squats and turns with the operator, waits after a robot reset"] if a.whole_body else [])
        print(f"[PASS] teleop hand path ({a.hand_mapping} mapping, {a.view} view{f', {len(warnings)} warning(s)' if warnings else ''}): waits for hands (not controllers), countdown, "
              "arms and fingers follow, " + ", ".join(checked + ["glitch ignored", "hold on tracking loss", "speed limits"]))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
