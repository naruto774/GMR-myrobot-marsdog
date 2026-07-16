import numpy as np
from general_motion_retargeting.utils.dog_bvh import load_dog_bvh_file
from general_motion_retargeting.utils.marsdog_axis import apply_marsdog_axis_correction_to_frames

def unit(v):
    return v / (np.linalg.norm(v) + 1e-8)

frames, _, _ = load_dog_bvh_file("assets/bvh/3331-clean.bvh")
frames = apply_marsdog_axis_correction_to_frames(frames)

pose = frames[0]

forward = unit(pose["back3"][0] - pose["back1"][0])
left = unit(pose["L_hip"][0] - pose["R_hip"][0])
up = unit(np.cross(forward, left))

print("forward =", forward)
print("left    =", left)
print("up      =", up)
print("dot forward +X =", forward @ np.array([1,0,0]))
print("dot left +Y    =", left @ np.array([0,1,0]))
print("dot up +Z      =", up @ np.array([0,0,1]))
