import mujoco
import numpy as np
from general_motion_retargeting.params import ROBOT_XML_DICT

model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["marsdog"]))
data = mujoco.MjData(model)
data.qpos[:] = 0.0          # 零位姿
mujoco.mj_forward(model, data)

def body_pos(name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[bid].copy()

head = body_pos("head_pitch_link")
tail_root = body_pos("tail1_virtual_link")   # 尾巴根部关节
tail1 = body_pos("tail1_link")               # 第一节连杆原点

print("3D distance (head -> tail root):", np.linalg.norm(head - tail_root))
print("3D distance (head -> tail1):", np.linalg.norm(head - tail1))

# 若只关心前后方向体长（base_link x 方向投影）
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
R_base = data.xmat[base_id].reshape(3, 3)
delta = head - tail_root
print("Projected body length (x):", abs(delta @ R_base[:, 0]))