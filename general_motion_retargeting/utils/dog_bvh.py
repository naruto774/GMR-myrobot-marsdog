import re

import numpy as np
from scipy.spatial.transform import Rotation as R

import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh


def _read_bvh_fps(bvh_file):
    with open(bvh_file, "r") as f:
        for line in f:
            match = re.search(r"Frame Time:\s*([0-9.]+)", line)
            if match:
                return round(1.0 / float(match.group(1)))
    return 30


def load_dog_bvh_file(bvh_file, robot_body_length=0.50):
    """
    Load quadruped dog BVH motion for GMR retargeting.

    Returns:
        frames: list[dict[str, tuple[np.ndarray, np.ndarray]]]
            Each frame maps bone name -> (position, orientation_wxyz).
        actual_body_length: target robot body length in meters (for scaling).
        fps: motion capture frame rate parsed from the BVH header.
    """
    data = read_bvh(bvh_file)
    global_data = utils.quat_fk(data.quats, data.pos, data.parents)

    rotation_matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    rotation_quat = R.from_matrix(rotation_matrix).as_quat(scalar_first=True)

    frames = []
    for frame in range(data.pos.shape[0]):
        result = {}
        for i, bone in enumerate(data.bones):
            orientation = utils.quat_mul(rotation_quat, global_data[0][frame, i])
            position = global_data[1][frame, i] @ rotation_matrix.T / 100.0
            result[bone] = (position.copy(), orientation.copy())
        frames.append(result)

    root_offset = frames[0]["root"][0].copy()
    for frame in frames:
        for bone in frame:
            pos, quat = frame[bone]
            frame[bone] = (pos - root_offset, quat)

    fps = _read_bvh_fps(bvh_file)
    return frames, robot_body_length, fps
