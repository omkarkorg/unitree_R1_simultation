NUM_MOTOR_SLOTS = 35

BODY_JOINT_SLOTS = {
    "left_hip_pitch_joint": 0,
    "left_hip_roll_joint": 1,
    "left_hip_yaw_joint": 2,
    "left_knee_joint": 3,
    "left_ankle_pitch_joint": 4,
    "left_ankle_roll_joint": 5,
    "right_hip_pitch_joint": 6,
    "right_hip_roll_joint": 7,
    "right_hip_yaw_joint": 8,
    "right_knee_joint": 9,
    "right_ankle_pitch_joint": 10,
    "right_ankle_roll_joint": 11,
    "waist_roll_joint": 12,
    "waist_yaw_joint": 13,
    "left_shoulder_pitch_joint": 15,
    "left_shoulder_roll_joint": 16,
    "left_shoulder_yaw_joint": 17,
    "left_elbow_joint": 18,
    "left_wrist_roll_joint": 19,
    "right_shoulder_pitch_joint": 22,
    "right_shoulder_roll_joint": 23,
    "right_shoulder_yaw_joint": 24,
    "right_elbow_joint": 25,
    "right_wrist_roll_joint": 26,
    "head_pitch_joint": 29,
    "head_yaw_joint": 30,
}

HAND_NUM_MOTORS = 6
HAND_MOTOR_JOINTS = ("thumb_metacarpal", "thumb_proximal", "index_proximal", "middle_proximal",
                     "ring_proximal", "pinky_proximal")
HAND_MOTOR_RANGES = (1.5184, 1.0472, 1.4661, 1.4661, 1.4661, 1.4661)
HAND_COUPLED_JOINTS = {
    "thumb_distal": (1, 1.0),
    "index_distal": (2, 1.155),
    "middle_distal": (3, 1.155),
    "ring_distal": (4, 1.155),
    "pinky_distal": (5, 1.155),
}
HAND_SIDES = ("left", "right")


def body_joint_names():
    return [name for name, _ in sorted(BODY_JOINT_SLOTS.items(), key=lambda kv: kv[1])]


def hand_joint_names(side):
    return [f"{side}_{j}_joint" for j in (*HAND_MOTOR_JOINTS, *HAND_COUPLED_JOINTS)]


def hand_motor_to_joint_positions(q_norm):
    motors = [float(q) * r for q, r in zip(q_norm, HAND_MOTOR_RANGES)]
    return motors + [ratio * motors[i] for i, ratio in HAND_COUPLED_JOINTS.values()]


def all_joint_names():
    return body_joint_names() + hand_joint_names("left") + hand_joint_names("right")
