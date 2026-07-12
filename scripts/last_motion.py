from general_motion_retargeting.utils.dog_bvh import load_dog_bvh_file
from general_motion_retargeting.utils.marsdog_axis import (
    MARSDOG_AXIS_CORRECTION_QUAT_WXYZ,
    apply_marsdog_axis_correction_to_frames,
)
import numpy as np
from scipy.spatial.transform import Rotation as R

bvh_path = "assets/bvh/moban1.bvh"
frames, body_length, fps = load_dog_bvh_file(bvh_path)
frames = apply_marsdog_axis_correction_to_frames(frames)

last_idx = len(frames) - 1          # 1036
pose = frames[last_idx]             # dict: bone -> (pos, quat_wxyz)

print(f"marsdog_axis_correction_wxyz={MARSDOG_AXIS_CORRECTION_QUAT_WXYZ}")

# 打印 IK 映射里关心的骨骼
ik_bones = [
    "root", "NewBone_2", "NewBone_7", "NewBone_12", "NewBone_21",
    "Should_L", "NewBone_13", "NewBone_16",
    "Should_R", "NewBone_17", "NewBone_20",
    "hip_L", "NewBone_3", "NewBone_4",
    "hip_R", "NewBone_8", "NewBone_9",
]
for name in ik_bones:
    if name in pose:
        p, q = pose[name]
        euler = R.from_quat(q, scalar_first=True).as_euler('xyz', degrees=True)
        print(f"{name}: pos={p}, quat(wxyz)={q}, euler(deg)={euler}")