import mujoco
import numpy as np
from general_motion_retargeting.params import ROBOT_XML_DICT

model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["marsdog"]))
data = mujoco.MjData(model)

# 站立构型：按你实际站立关节角设置 qpos
# data.qpos[:3] = base 位置, [3:7] = base 四元数 (wxyz in MuJoCo)
# data.qpos[7:] = 各关节角
mujoco.mj_forward(model, data)

robot_links = [
    "base_link", "waist_yaw_link", "waist_pitch_link",
    "neck_pitch_link", "head_pitch_link",
    "rl_thigh_link", "rl_calf_link",
    "fl_hip_pitch_link", "fl_calf_link", "fl_foot_link",
    "rr_thigh_link", "rr_calf_link", 
    "fr_hip_pitch_link", "fr_calf_link", "fr_foot_link",
    # ... 按 bvh_to_marsdog.json 补全
]
for link in robot_links:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
    p = data.xpos[bid].copy()
    q = data.xquat[bid].copy()   # MuJoCo 也是 wxyz
    print(f"{link}: pos={p}, quat={q}")