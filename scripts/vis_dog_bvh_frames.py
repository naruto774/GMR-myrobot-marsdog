"""Visualize GMR-loaded dog BVH joint positions and global frames.

Default setup matches bvh_to_dison.json (rom/dison skeleton naming).

Example:

    python scripts/vis_dog_bvh_frames.py
    python scripts/vis_dog_bvh_frames.py --bvh_file assets/bvh/rom_02_T_180.bvh
    python scripts/vis_dog_bvh_frames.py --bvh_file assets/bvh/1121-clean.bvh --frame 0
"""

import argparse
from pathlib import Path
import sys
import time

import mujoco
import numpy as np
from mujoco import viewer
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting.utils.dog_bvh import (  # noqa: E402
    _resolve_root_name,
    load_dog_bvh_file,
)
from general_motion_retargeting.utils.marsdog_axis import (  # noqa: E402
    apply_marsdog_axis_correction_to_frames,
)
from general_motion_retargeting.utils.lafan_vendor.extract import read_bvh  # noqa: E402


# ---- Defaults (override via CLI) ------------------------------------------
BVH_FILE = REPO_ROOT / "assets/bvh/3331-clean.bvh"
FRAME_IDX = 0
APPLY_MARSDOG_AXIS = True

# Bones listed in bvh_to_dison.json ik_match_table1.
FOCUS_BONES = {
    "root",
    "back1",
    "back2",
    "back3",
    "neck1",
    "head",
    "L_arm",
    "L_forearm",
    "L_hand",
    "R_arm",
    "R_forearm",
    "r_hand",
    "L_hip",
    "L_leg",
    "L_foot",
    "R_hip",
    "R_leg",
    "R_foot",
}

SHOW_ALL_FRAMES = False
SHOW_ALL_NAMES = False
FRAME_SIZE = 0.08
FOCUS_FRAME_SIZE = 0.22
JOINT_RADIUS = 0.018
FOCUS_JOINT_RADIUS = 0.035
BONE_WIDTH = 0.008
# ---------------------------------------------------------------------------


EMPTY_WORLD_XML = """
<mujoco model="dog_bvh_frames">
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <worldbody>
    <geom name="ground" type="plane" size="10 10 0.01" rgba="0.2 0.2 0.2 0.25"/>
  </worldbody>
</mujoco>
"""


def normalize_quat(quat):
    quat = np.asarray(quat, dtype=float)
    return quat / np.linalg.norm(quat)


def add_sphere(scene, pos, radius, rgba, label=None):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0.0, 0.0],
        pos=np.asarray(pos, dtype=float),
        mat=np.eye(3).reshape(-1),
        rgba=rgba,
    )
    if label is not None:
        geom.label = label
    scene.ngeom += 1


def add_segment(scene, p0, p1, width, rgba):
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        size=[width, 0.0, 0.0],
        pos=np.zeros(3),
        mat=np.eye(3).reshape(-1),
        rgba=rgba,
    )
    mujoco.mjv_connector(
        geom,
        type=mujoco.mjtGeom.mjGEOM_CAPSULE,
        width=width,
        from_=np.asarray(p0, dtype=float),
        to=np.asarray(p1, dtype=float),
    )
    scene.ngeom += 1


def add_frame(scene, pos, mat, size, label=None):
    colors = (
        [1.0, 0.0, 0.0, 1.0],  # x red
        [0.0, 1.0, 0.0, 1.0],  # y green
        [0.0, 0.25, 1.0, 1.0],  # z blue
    )
    for axis_idx, color in enumerate(colors):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            size=[0.01, 0.01, 0.01],
            pos=np.asarray(pos, dtype=float),
            mat=np.asarray(mat, dtype=float).reshape(-1),
            rgba=color,
        )
        if label is not None and axis_idx == 0:
            geom.label = label
        mujoco.mjv_connector(
            geom,
            type=mujoco.mjtGeom.mjGEOM_ARROW,
            width=0.006,
            from_=np.asarray(pos, dtype=float),
            to=np.asarray(pos, dtype=float) + size * mat[:, axis_idx],
        )
        scene.ngeom += 1


def print_bone_report(pose, bone_name):
    pos, quat = pose[bone_name]
    mat = R.from_quat(normalize_quat(quat), scalar_first=True).as_matrix()
    print(f"\n[{bone_name}]")
    print(f"pos       = {np.array2string(np.asarray(pos), precision=6)}")
    print(f"quat wxyz = {np.array2string(normalize_quat(quat), precision=6)}")
    print(f"x_axis    = {np.array2string(mat[:, 0], precision=6)}")
    print(f"y_axis    = {np.array2string(mat[:, 1], precision=6)}")
    print(f"z_axis    = {np.array2string(mat[:, 2], precision=6)}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize dog BVH frames for bvh_to_dison retargeting."
    )
    parser.add_argument("--bvh_file", type=str, default=str(BVH_FILE))
    parser.add_argument("--frame", type=int, default=FRAME_IDX)
    parser.add_argument(
        "--no-marsdog-axis",
        action="store_true",
        help="Disable Marsdog global axis correction.",
    )
    parser.add_argument(
        "--show-all-frames",
        action="store_true",
        default=SHOW_ALL_FRAMES,
        help="Draw RGB frames for every bone, not only FOCUS_BONES.",
    )
    parser.add_argument(
        "--show-all-names",
        action="store_true",
        default=SHOW_ALL_NAMES,
        help="Label every joint sphere.",
    )
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def main():
    args = parse_args()
    bvh_file = resolve_path(args.bvh_file)
    apply_marsdog_axis = not args.no_marsdog_axis

    frames, _, fps = load_dog_bvh_file(str(bvh_file))
    if apply_marsdog_axis:
        frames = apply_marsdog_axis_correction_to_frames(frames)
    frame_idx = args.frame if args.frame >= 0 else len(frames) - 1
    pose = frames[frame_idx]
    root_name = _resolve_root_name(frames)

    data = read_bvh(str(bvh_file))
    bones = list(data.bones)
    parents = list(data.parents)

    print("=== Dog BVH Frame Visualization (bvh_to_dison) ===")
    print(f"bvh_file = {bvh_file}")
    print("Pipeline: load_dog_bvh_file() -> Y-up to Z-up + root XY recenter")
    if apply_marsdog_axis:
        print("           -> apply_marsdog_axis_correction_to_frames()")
    print("RGB: x=red, y=green, z=blue")
    print(f"fps   = {fps}")
    print(f"frame = {frame_idx} / {len(frames) - 1}")
    print(f"root  = {root_name}")
    for bone_name in sorted(FOCUS_BONES):
        if bone_name in pose:
            print_bone_report(pose, bone_name)
        else:
            print(f"\n[warn] focus bone not in pose: {bone_name}")

    model = mujoco.MjModel.from_xml_string(EMPTY_WORLD_XML)
    mj_data = mujoco.MjData(model)

    with viewer.launch_passive(model, mj_data, show_left_ui=False, show_right_ui=False) as v:
        root_pos = pose[root_name][0]
        v.cam.lookat[:] = root_pos
        v.cam.distance = 3.0
        v.cam.elevation = -20
        v.cam.azimuth = 140

        while v.is_running():
            v.user_scn.ngeom = 0

            for child_idx, parent_idx in enumerate(parents):
                if parent_idx < 0:
                    continue
                child_name = bones[child_idx]
                parent_name = bones[parent_idx]
                if child_name in pose and parent_name in pose:
                    add_segment(
                        v.user_scn,
                        pose[parent_name][0],
                        pose[child_name][0],
                        BONE_WIDTH,
                        [0.85, 0.85, 0.85, 0.75],
                    )

            for bone_name, (pos, quat) in pose.items():
                is_focus = bone_name in FOCUS_BONES
                add_sphere(
                    v.user_scn,
                    pos,
                    FOCUS_JOINT_RADIUS if is_focus else JOINT_RADIUS,
                    [1.0, 0.75, 0.1, 1.0] if is_focus else [0.7, 0.7, 0.7, 0.8],
                    label=bone_name if (is_focus or args.show_all_names) else None,
                )

                if args.show_all_frames or is_focus:
                    mat = R.from_quat(normalize_quat(quat), scalar_first=True).as_matrix()
                    add_frame(
                        v.user_scn,
                        pos,
                        mat,
                        FOCUS_FRAME_SIZE if is_focus else FRAME_SIZE,
                        label=f"frame:{bone_name}" if is_focus else None,
                    )

            v.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
