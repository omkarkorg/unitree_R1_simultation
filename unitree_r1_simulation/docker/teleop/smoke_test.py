#!/usr/bin/env python3

import sys

import numpy as np

sys.path.insert(0, "/opt/xr_teleoperate")

import casadi
import cv2
import logging_mp
import meshcat
import nlopt
import pinocchio as pin
import rerun
import sshkeyboard
import teleimager.image_client
import televuer
import torch
import unitree_sdk2py
import vuer
import zmq
from pinocchio import casadi as cpin

assert np.__version__ == "1.26.4", np.__version__
print(f"imports OK: pinocchio {pin.__version__}, casadi {casadi.__version__}, cv2 {cv2.__version__}, "
      f"torch {torch.__version__}")

from teleop.robot_control.robot_arm_ik import R1_A5_ArmIK

ik = R1_A5_ArmIK()
model = ik.reduced_robot.model
data = model.createData()
assert model.nq == 10, model.nq
q_ref = np.array([-0.4, 0.3, 0.1, 0.9, 0.2, -0.4, -0.3, -0.1, 0.9, -0.2])
pin.framesForwardKinematics(model, data, q_ref)
left = data.oMf[ik.L_hand_id].homogeneous.copy()
right = data.oMf[ik.R_hand_id].homogeneous.copy()
q_sol, tau = ik.solve_ik(left, right, np.zeros(10), np.zeros(10))
assert q_sol is not None and q_sol.shape == (10,) and tau.shape == (10,)
for _ in range(20):
    q_sol, tau = ik.solve_ik(left, right, q_sol, np.zeros(10))
pin.framesForwardKinematics(model, data, q_sol)
err = max(np.linalg.norm(data.oMf[ik.L_hand_id].translation - left[:3, 3]),
          np.linalg.norm(data.oMf[ik.R_hand_id].translation - right[:3, 3]))
print(f"R1_A5 IK OK: position error {err * 1000:.1f} mm, q = {np.round(q_sol, 3)}")
assert err < 0.02, f"IK position error too large: {err}"

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

hand = HandRetargeting(HandType.BRAINCO_HAND)
expected = ["left_thumb_metacarpal_joint", "left_thumb_proximal_joint", "left_index_proximal_joint",
            "left_middle_proximal_joint", "left_ring_proximal_joint", "left_pinky_proximal_joint"]
assert hand.left_brainco_api_joint_names == expected, hand.left_brainco_api_joint_names
print("BrainCo retargeting OK:", hand.left_retargeting_joint_names)

import py_compile

main_script = "/opt/xr_teleoperate/teleop/teleop_hand_and_arm.py"
py_compile.compile(main_script, doraise=True)
patched = open(main_script).read()
assert "--hand-mapping" in patched and "--head-tracking" in patched and "--vr-buttons" in patched \
    and "--whole-body" in patched, "docker/teleop/patches/r1_teleop.patch not applied"
from teleop.robot_control.robot_arm import R1_A5_ArmController
assert hasattr(R1_A5_ArmController, "ctrl_head"), "r1_teleop.patch: R1_A5_ArmController.ctrl_head missing"
from televuer import TeleVuer
assert hasattr(TeleVuer, "on_controller_watch") and hasattr(TeleVuer, "main_lab"), "docker/teleop/patches/televuer_r1.patch not applied"
print("R1 teleop patches OK")

sys.path.insert(0, "/opt/r1_sim")
from r1_sim.layout import lab_description
from teleop.robot_control.robot_hand_brainco import BraincoDirectMapping
from televuer.lab_view import LabScene

src = open(main_script).read().split("if __name__ == '__main__':")[0].replace("logging_mp.basicConfig(level=logging_mp.INFO)", "")
helpers = {"__name__": "smoke", "__file__": main_script}
exec(compile(src, main_script, "exec"), helpers)
geometry = helpers["RobotGeometry"](helpers["R1_MODEL_URDF"], [0.095, 0.0, 0.0])
assert np.allclose(geometry.ik_origin, [0.0325, 0.0, 0.049], atol=1e-4), geometry.ik_origin
ik_palm = R1_A5_ArmIK(ee_placement=geometry.palm_in_wrist, cost_weights=(5000.0, 1.0, 0.02, 0.1), filter_weights=(1.0,))
m2 = ik_palm.reduced_robot.model
d2 = m2.createData()
pin.framesForwardKinematics(m2, d2, q_ref)
left, right = d2.oMf[ik_palm.L_hand_id].homogeneous.copy(), d2.oMf[ik_palm.R_hand_id].homogeneous.copy()
q_sol = np.zeros(10)
for _ in range(20):
    q_sol, _ = ik_palm.solve_ik(left, right, q_sol, np.zeros(10))
pin.framesForwardKinematics(m2, d2, q_sol)
err = max(np.linalg.norm(d2.oMf[ik_palm.L_hand_id].translation - left[:3, 3]),
          np.linalg.norm(d2.oMf[ik_palm.R_hand_id].translation - right[:3, 3]))
print(f"R1_A5 IK with the BrainCo palm as end effector OK: position error {err * 1000:.2f} mm")
assert err < 0.002, f"palm IK position error too large: {err}"
mapping = BraincoDirectMapping(helpers["R1_MODEL_URDF"])
print("direct finger mapping OK, robot hand angles (deg) open..closed:", mapping.ranges("right"))
lab = LabScene(lab_description(0.58, 0.75, True), helpers["R1_MODEL_URDF"], helpers["R1_VR_MESHES"])
poses = lab.link_poses(np.zeros(len(lab.q_index)))
faces = sum(len(f) for _, _, _, f in lab.items)
assert len(poses) > 50 and 30000 < faces < 120000, (len(poses), faces)
print(f"lab view OK: {len(poses)} robot links, {faces} triangles, {len(lab.q_index)} joints")

desc = lab_description(0.58, 0.725, True, "assembly", True)
lab = LabScene(desc, helpers["R1_MODEL_URDF"], helpers["R1_VR_MESHES"])
state = {"q": [0.0] * len(lab.q_index), "root": [0.0, 0.0, 0.725, 0, 0, 0, 1],
         "objects": {o["name"]: [0.0, 0.0, 0.5, 0, 0, 0, 1] for o in desc["objects"]}}
group = lab.static_nodes(np.eye(4), state)
keys = [c.key for c in group.children]
assert all(f"lab-{m['name']}" in keys for m in desc["meshes"]) and "obj-nut" in keys and "lab-side_table" in keys, keys
lab.robot_nodes(state["q"])
moved = lab.updates({**state, "root": [1.0, 0.5, 0.6, 0, 0, 0, 1]})
pelvis = [n for n in moved if n["key"] == "link-pelvis_link"]
assert pelvis and abs(pelvis[0]["matrix"][12] - 1.0) < 1e-6 and abs(pelvis[0]["matrix"][13] - 0.5) < 1e-6, moved[:2]
print(f"lab view (assembly, free-standing) OK: {len(desc['meshes'])} fixtures, {len(desc['objects'])} parts, "
      f"robot drawn at the streamed pelvis pose")

from r1_sim import loco_spec
BF = helpers["BodyFollower"]
assert (BF.H_NOM, BF.H_MIN, BF.PITCH_MAX) == (loco_spec.H_NOM, loco_spec.USE_H_MIN, loco_spec.PITCH_MAX), \
    "BodyFollower's range differs from r1_sim/r1_sim/loco_spec.py"
assert loco_spec.VX_RANGE[0] <= BF.VX[0] < 0 < BF.VX[1] <= loco_spec.VX_RANGE[1] and \
    loco_spec.VY_RANGE[0] <= BF.VY[0] < 0 < BF.VY[1] <= loco_spec.VY_RANGE[1] and 0 < BF.WZ <= loco_spec.WZ_RANGE[1], \
    "BodyFollower's speeds exceed the leg controller's range"
assert 0.40 < geometry.reach < 0.47, f"arm reach {geometry.reach:.3f} m: 90 % of the stretched arm (0.46 m), not of the " \
    "zero pose (elbow bent 90 deg)"
sent = []
body = helpers["BodyFollower"](geometry, send=sent.append)
root = np.eye(4); root[:3, 3] = [0.0, 0.0, 0.725]
eye = root[:3, 3] + geometry.eye
def head_at(p, yaw=0.0):
    H = np.eye(4); H[:3, :3] = helpers["rot_z"](yaw); H[:3, 3] = p
    return H
body.start(root, eye[2])
cmd = [body.step(t * 0.033, head_at(eye), None, root, True) for t in range(30)][-1]
assert abs(cmd[0]) < 1e-9 and abs(cmd[1]) < 1e-9 and abs(cmd[2]) < 1e-9 and cmd[3] > 0.70 and cmd[4] < 0.05, cmd
cmd = body.step(1.1, head_at(eye + [0.05, 0.03, 0.0]), None, root, True)
assert cmd[0] == 0.0 and cmd[1] == 0.0, f"a 6 cm head movement must not make the robot step: {cmd}"
cmd = body.step(1.2, head_at(eye + [0.50, 0.0, 0.0]), None, root, True)
assert cmd[0] > 0.5 and abs(cmd[1]) < 0.05, f"walking forward: {cmd}"
body.start(root, eye[2])
cmd = [body.step(2.0 + t * 0.033, head_at(eye - [0.0, 0.0, 0.20]), None, root, True) for t in range(60)][-1]
assert cmd[3] < body.H_MIN + 0.01 and abs(cmd[0]) < 1e-9, f"squatting 20 cm: {cmd}"
body.start(root, eye[2])
low_palm = np.eye(4); low_palm[:3, 3] = [0.30, 0.15, 0.30]
cmd = [body.step(5.0 + t * 0.033, head_at(eye), [low_palm, None], root, True) for t in range(60)][-1]
assert cmd[3] < 0.66 or cmd[4] > 0.25, f"a low hand target must make the robot squat or lean: {cmd}"
body.start(root, eye[2])
cmd = [body.step(8.0 + t * 0.033, head_at(eye, yaw=np.radians(20.0)), None, root, True) for t in range(5)][-1]
assert cmd[2] == 0.0, f"looking 20 deg aside must not turn the robot: {cmd}"
cmd = body.step(8.3, head_at(eye, yaw=np.radians(70.0)), None, root, True)
assert cmd[2] > 0.5, f"turning the body 70 deg: {cmd}"
assert sent and set(sent[-1]) == {"vx", "vy", "wz", "height", "pitch"}, sent[-1]
print(f"whole-body mapping OK: still within the leash, walks, squats, bends for a low hand, turns past "
      f"{np.degrees(body.YAW_LEASH):.0f} deg (arm reach {geometry.reach:.2f} m)")
body.stand_up()
assert sent[-1] == {"vx": 0.0, "vy": 0.0, "wz": 0.0, "height": body.H_NOM, "pitch": 0.0}, sent[-1]
assert body.upright(root) and not body.upright(head_at([0.0, 0.0, 0.62]))

sent.clear()
pb = helpers["BodyFollower"](geometry, send=sent.append, body={"mode": "exact", "h_min": 0.50, "h_nom": 0.725, "pitch_max": 0.55})
assert pb.pose_mode and pb.H_MIN == 0.50 and not body.pose_mode
pb.start(root, eye[2])
cmd = [pb.step(20.0 + t * 0.033, head_at(eye), None, root, True) for t in range(30)][-1]
assert abs(cmd[0]) < 1e-9 and abs(cmd[1]) < 1e-9 and abs(cmd[2]) < 1e-9 and cmd[3] > 0.70 and cmd[4] < 0.05, cmd
assert set(sent[-1]) == {"x", "y", "yaw", "height", "pitch"}, sent[-1]
cmd = pb.step(21.1, head_at(eye + [0.05, 0.03, 0.0]), None, root, True)
assert cmd[0] == 0.0 and cmd[1] == 0.0, f"pose mode: a 6 cm head movement must not move the pelvis: {cmd}"
cmd = pb.step(21.2, head_at(eye + [0.50, 0.0, 0.0]), None, root, True)
assert 0.0 < cmd[0] <= pb.V_MAX * 0.1 + 1e-9, f"pose mode: the pose sent moves as fast as the simulator places it: {cmd}"
cmd = [pb.step(21.2 + t * 0.033, head_at(eye + [0.50, 0.0, 0.0]), None, root, True) for t in range(1, 30)][-1]
assert abs(cmd[0] - (0.50 - pb.LEASH)) < 1e-6 and abs(cmd[1]) < 1e-6, f"pose mode: walking 0.5 m: {cmd}"
T = pb.pelvis()
from r1_sim.kinematic_body import pelvis_matrix
assert np.allclose(T, pelvis_matrix(cmd[0], cmd[1], cmd[3], cmd[2], cmd[4])), "pose mode: pelvis() is not the pose the simulator places"
def nodding(pitch, neck):
    H = np.eye(4); c, s_ = np.cos(pitch), np.sin(pitch)
    H[:3, :3] = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])
    H[:3, 3] = neck + H[:3, :3] @ pb.NECK_TO_EYE
    return H
neck0 = eye - pb.NECK_TO_EYE
work = [np.eye(4), np.eye(4)]
work[0][:3, 3] = root[:3, 3] + geometry.shoulder["left"] + [0.30, 0.03, -0.25]
work[1][:3, 3] = root[:3, 3] + geometry.shoulder["right"] + [0.30, -0.03, -0.25]
rng = np.random.default_rng(1)
pb.start(root, eye[2], head=nodding(np.radians(30.0), neck0))
cmds = []
for t in range(300):
    sway = rng.uniform(-0.01, 0.01, 3)
    hands = [P.copy() for P in work]
    for P in hands:
        P[:3, 3] += 0.03 * np.sin(t / 15.0)
    cmds.append(pb.step(30.0 + t * 0.033, nodding(np.radians(30.0 + 30.0 * np.sin(t / 40.0)), neck0 + sway), hands, root, True))
cmds = np.array(cmds)
assert np.ptp(cmds[:, 0]) == 0.0 and np.ptp(cmds[:, 1]) == 0.0 and np.ptp(cmds[:, 2]) == 0.0, "standing at the table: it stepped or turned"
assert cmds[:, 3].min() > pb.H_NOM - 0.005 and cmds[:, 4].max() < 0.03, \
    f"standing at the table, looking down, hands within reach: the robot bent (height {cmds[:, 3].min():.3f}, lean {cmds[:, 4].max():.3f})"
lean = [pb.step(41.0 + t * 0.033, nodding(0.5, neck0 + [0.12, 0.0, -0.03]), work, root, True) for t in range(60)][-1]
assert lean[4] > 0.15, f"pose mode: leaning the upper body forward 12 cm must lean the robot: {lean}"
pb.start(root, eye[2], head=nodding(np.radians(45.0), neck0))
cmd = [pb.step(45.0 + t * 0.033, nodding(np.radians(45.0), neck0 - [0.0, 0.0, 0.15]), work, root, True) for t in range(60)][-1]
assert abs(cmd[3] - (pb.H_NOM - 0.15)) < 0.02 and cmd[4] < 0.05, f"pose mode: squatting 15 cm after GO looking down: {cmd}"
pb.start(root, eye[2])
cmd = [pb.step(22.0 + t * 0.033, head_at(eye - [0.0, 0.0, 0.20]), None, root, True) for t in range(60)][-1]
assert abs(cmd[3] - (0.725 - 0.20)) < 0.03 and abs(cmd[0]) < 1e-9, f"pose mode: squatting 20 cm (to below the balance range): {cmd}"
before = list(pb.command)
pb.stand()
assert pb.command == before, "pose mode: standing still keeps the pose"
moved = root.copy(); moved[:3, 3] = [0.3, -0.2, 0.725]
pb.stand_up(moved)
assert pb.command[:2] == [0.3, -0.2] and pb.command[3] == pb.H_NOM and pb.command[4] == 0.0, pb.command
op = helpers["OperatorFrame"].__new__(helpers["OperatorFrame"])
op.heading, op.eye_xr, op.world_scale, op.robot_eye_z = np.eye(3), np.array([0.3, 1.62, -0.4]), None, 0.725 + geometry.eye[2]
eye_sim = np.array([0.0, 0.0, 0.725]) + geometry.eye
T = op.placement(eye_sim)
assert abs(op.scale - 1.62 / eye_sim[2]) < 1e-9 and abs((T @ [0.2, 0.1, 0.0, 1.0])[1]) < 1e-9, "the robot's floor is not the real floor"
assert np.allclose(T @ np.append(eye_sim, 1.0), np.append(op.eye_xr, 1.0)), "the robot's eyes are not at the operator's"
P = np.eye(4); P[:3, :3] = helpers["rot_z"](0.7) @ helpers["rot_y"](0.4); P[:3, 3] = [0.5, 1.1, -0.2]
Q = helpers["from_room"](T, P)
assert np.allclose(Q[:3, :3] @ Q[:3, :3].T, np.eye(3)) and np.allclose(helpers["to_room"](T, Q), P), "from_room / to_room"
op.world_scale = 1.0
assert op.placement(eye_sim) is not None and op.scale == 1.0
buttons = helpers["VRButtons"]()
buttons.place(T, eye_sim)
for pose in buttons.poses.values():
    assert np.allclose(pose[:3, :3] @ pose[:3, :3].T, np.eye(3)) and np.linalg.norm(pose[:3, 3] - op.eye_xr) < 0.95, pose
ps = helpers["BodyFollower"](geometry, send=sent.append, body={"mode": "exact", "h_min": 0.50, "h_nom": 0.725, "pitch_max": 0.55})
s_ = op.scale = 1.4
ps.start(root, eye[2], head=head_at(eye), scale=s_)
cmd = [ps.step(50.0 + t * 0.033, head_at(eye - [0.0, 0.0, 0.14 / s_]), None, root, True) for t in range(60)][-1]
assert abs(cmd[3] - (ps.H_NOM - 0.14 / s_)) < 0.02, f"scaled world: a 14 cm squat in the room is 10 cm of the robot: {cmd}"
print("world scale OK: robot's floor on the real floor, eyes at the operator's, poses stay poses, buttons within reach")
print("whole-body mapping, placed body OK: pelvis pose follows the operator (leash, at the simulator's speed), squats to "
      "0.50 m, leans with the upper body, holds; looking down, swaying and working hands within reach do not move it")

from televuer.lab_view import panel_image
VRButtons = helpers["VRButtons"]
assert list(VRButtons.ANCHORS) == ["exit", "reset", "restart", "start", "rec"], list(VRButtons.ANCHORS)
buttons = VRButtons()
buttons.place(np.eye(4), eye)
t = 0.0
for name, T in buttons.poses.items():
    c, n = T[:3, 3], T[:3, 2]
    got = []
    for d in (0.10, 0.06, 0.02, -0.005, -0.03):
        t += 0.05
        got += buttons.step(t, [c + d * n])
    assert got == [name], (name, got)
    for sign in (1.0, -1.0):
        edge = c + T[:3, 1] * (VRButtons.SIZE[1] / 2 + VRButtons.MARGIN) * sign
        for other, U in buttons.poses.items():
            rel = edge - U[:3, 3]
            inside = abs(rel @ U[:3, 0]) <= VRButtons.SIZE[0] / 2 + VRButtons.MARGIN and abs(rel @ U[:3, 1]) <= VRButtons.SIZE[1] / 2 + VRButtons.MARGIN
            assert other == name or not inside, f"the press zones of {name} and {other} overlap"
    t += 2.0
assert not buttons.confirm(10.0, "exit") and buttons.confirm(11.5, "exit"), "EXIT: the second press within 4 s confirms"
assert not buttons.confirm(12.0, "reset") and not buttons.confirm(16.5, "reset"), "RESET: a second press after 4 s arms again"
labels = helpers["button_labels"](0.0, {}, False, True, False, 0.0)
assert labels["start"][2] and not labels["restart"][2] and labels["reset"][0] == "RESET" and labels["exit"][2], labels
labels = helpers["button_labels"](1.0, {"exit": 4.0}, True, True, True, 75.0)
assert labels["exit"][0][0] == ("EXIT?", 1.0) and labels["restart"][2] and not labels["start"][2] and labels["rec"][0] == "SAVE 1:15", labels
for spec in buttons.panels(1.0, labels):
    assert panel_image(spec).shape[2] == 3
head = np.eye(4); head[:3, 3] = [0.0, 1.6, 0.0]
end = helpers["ended_panel"](head)
M = np.array(end["matrix"]).reshape(4, 4, order="F")
assert np.allclose(M[:3, 3], [0.0, 1.45, -0.8]) and np.allclose(M[:3, 2], [0.0, 0.0, 1.0]) and np.isclose(np.linalg.det(M[:3, :3]), 1.0), M
panel_image(end)
print("VR buttons OK: EXIT, RESET, RESTART, START, REC; presses reach only their button; RESET / EXIT confirmed by a second press")
print("SMOKE TEST PASSED")
