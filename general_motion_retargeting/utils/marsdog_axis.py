"""Marsdog-specific axis correction for dog BVH retargeting.

The dog BVH loader is shared by multiple quadruped paths. Marsdog expects
MuJoCo's semantic world axes: +X forward, +Y left, +Z up. The dog BVH data used
here arrives with dog-forward near -Y, dog-left near +X, and dog-up near +Z
after the common loader. This module keeps that global axis fix separate from
per-link rot_offset calibration.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

#marsdog的坐标系转换矩阵
MARSDOG_AXIS_CORRECTION_MATRIX = np.array(  
    [
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)

# MARSDOG_AXIS_CORRECTION_MATRIX = np.array(  
#     [
#         [0.0, 0.0, -1.0],
#         [-1.0, 0.0, 0.0],
#         [0.0, 1.0, 0.0],
#     ]
# )
MARSDOG_AXIS_CORRECTION = R.from_matrix(MARSDOG_AXIS_CORRECTION_MATRIX) #将矩阵转换为四元数
MARSDOG_AXIS_CORRECTION_QUAT_WXYZ = MARSDOG_AXIS_CORRECTION.as_quat( #将四元数转换为wxyz格式
    scalar_first=True
)


def apply_marsdog_axis_correction_to_pose(pose):  
    """将pose的坐标系转换为marsdog的坐标系+x forward, +y left, +z up"""
    corrected_pose = {} #存储转换后的pose
    for body_name, (pos, quat) in pose.items(): #遍历pose中的每个body
        corrected_pos = MARSDOG_AXIS_CORRECTION.apply(np.asarray(pos, dtype=float)) #将pos转换为marsdog的坐标系
        corrected_quat = ( #将quat转换为marsdog的坐标系
            MARSDOG_AXIS_CORRECTION
            * R.from_quat(np.asarray(quat, dtype=float), scalar_first=True)
        ).as_quat(scalar_first=True)
        corrected_pose[body_name] = (corrected_pos, corrected_quat)  #存储转换后的pose
    return corrected_pose #返回转换后的pose

#将frames的坐标系转换为marsdog的坐标系
def _resolve_root_name(frames, root_name=None):
    if root_name is not None:
        if root_name not in frames[0]:
            raise KeyError(f"Root bone not found in BVH pose: {root_name}")
        return root_name
    for candidate in ("Hips", "root"):
        if candidate in frames[0]:
            return candidate
    return next(iter(frames[0]))


def apply_marsdog_axis_correction_to_frames(
    frames, root_name=None, recenter_horizontal=True
):
    """Apply the Marsdog axis correction to every frame from load_dog_bvh_file.

    The shared BVH loader recenters horizontal motion before this Marsdog-only
    axis correction. After the axes are rotated, the corrected horizontal plane
    is different, so we recenter X/Y once more while preserving vertical height.
    """
    corrected_frames = [apply_marsdog_axis_correction_to_pose(frame) for frame in frames]
    if not recenter_horizontal or not corrected_frames:
        return corrected_frames
    root_name = _resolve_root_name(corrected_frames, root_name)
    # 1. 提取旋转后，第一帧根节点的水平位移偏移量
    root_offset = corrected_frames[0][root_name][0].copy()
    root_offset[2] = 0.0  # 保留高度，避免机器狗陷入地面
    # 2. 让所有帧的每一个刚体，都减去这个第一帧的水平偏移
    for frame in corrected_frames:
        for body_name, (pos, quat) in frame.items():
            frame[body_name] = (pos - root_offset, quat)
    return corrected_frames

