"""Inspect LAFAN1 BVH frame alignment against a G1 MuJoCo/URDF link frame.

This script focuses on the GMR convention:

    q_target = q_bvh_after_loader * q_offset
    q_offset = inverse(q_bvh_after_loader) * q_robot_link

It prints the world axes of a BVH bone and a robot link, then compares the
computed offset with the value stored in the GMR IK config.

Example:
    python scripts/check_lafan1_g1_frame_offset.py

    python scripts/check_lafan1_g1_frame_offset.py \
        --bvh_file assets/bvh/dance2_subject4.bvh \
        --robot_file assets/unitree_g1/g1_custom_collision_29dof.urdf \
        --bvh_bone LeftUpLeg \
        --robot_link left_hip_yaw_link \
        --frame 0
"""

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting.params import IK_CONFIG_ROOT  # noqa: E402
from general_motion_retargeting.utils.lafan1 import load_lafan1_file  # noqa: E402


def normalize_quat(quat):
    quat = np.asarray(quat, dtype=float)
    return quat / np.linalg.norm(quat)


def canonicalize_quat(quat):
    """Use a stable sign convention because q and -q encode the same rotation."""
    quat = normalize_quat(quat)
    return -quat if quat[0] < 0 else quat


def quat_to_matrix(quat_wxyz):
    return R.from_quat(normalize_quat(quat_wxyz), scalar_first=True).as_matrix()


def print_frame_axes(title, pos, quat_wxyz):
    mat = quat_to_matrix(quat_wxyz)
    print(f"\n[{title}]")
    print(f"pos        = {np.array2string(np.asarray(pos), precision=6)}")
    print(f"quat_wxyz  = {np.array2string(canonicalize_quat(quat_wxyz), precision=6)}")
    print("axes in world frame (matrix columns):")
    print(f"  x_axis -> {np.array2string(mat[:, 0], precision=6)}")
    print(f"  y_axis -> {np.array2string(mat[:, 1], precision=6)}")
    print(f"  z_axis -> {np.array2string(mat[:, 2], precision=6)}")
    print("rotation matrix:")
    print(np.array2string(mat, precision=6, suppress_small=True))


def load_config_offset(config_path, robot_link):
    with open(config_path, "r") as f:
        config = json.load(f)

    for table_name in ("ik_match_table1", "ik_match_table2"):
        table = config.get(table_name, {})
        if robot_link in table:
            return np.asarray(table[robot_link][4], dtype=float), table_name

    raise KeyError(f"{robot_link} not found in {config_path}")


def load_robot_link_pose(robot_file, robot_link):
    model = mujoco.MjModel.from_xml_path(str(robot_file))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot_link)
    if body_id < 0:
        names = []
        for i in range(model.nbody):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name:
                names.append(name)
        raise KeyError(
            f"Robot body not found: {robot_link}\n"
            f"Available bodies include: {names[:40]}"
        )

    return data.xpos[body_id].copy(), data.xquat[body_id].copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        type=Path,
        default=REPO_ROOT / "assets/bvh/dance2_subject4.bvh",
    )
    parser.add_argument(
        "--robot_file",
        type=Path,
        default=REPO_ROOT / "assets/unitree_g1/g1_custom_collision_29dof.urdf",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=IK_CONFIG_ROOT / "bvh_to_g1.json",
    )
    parser.add_argument("--bvh_bone", type=str, default="LeftUpLeg")
    parser.add_argument("--robot_link", type=str, default="left_hip_yaw_link")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="BVH frame index. Use -1 for the final frame.",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Scan all BVH frames and report frames where q_bvh * q_json is closest to q_robot.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
        help="Number of best frames to print when --scan is enabled.",
    )
    args = parser.parse_args()

    frames, _ = load_lafan1_file(str(args.bvh_file))
    if args.bvh_bone not in frames[0]:
        raise KeyError(
            f"BVH bone not found: {args.bvh_bone}\n"
            f"Available bones include: {list(frames[0].keys())[:40]}"
        )
    robot_pos, q_robot = load_robot_link_pose(args.robot_file, args.robot_link)
    q_json, table_name = load_config_offset(args.config, args.robot_link)

    r_robot = R.from_quat(normalize_quat(q_robot), scalar_first=True)
    r_json = R.from_quat(normalize_quat(q_json), scalar_first=True)

    if args.scan:
        errors = []
        for idx, pose in enumerate(frames):
            _, q_bvh_i = pose[args.bvh_bone]
            r_bvh_i = R.from_quat(normalize_quat(q_bvh_i), scalar_first=True)
            q_updated_i = (r_bvh_i * r_json).as_quat(scalar_first=True)
            err_deg = np.degrees(
                (
                    R.from_quat(q_updated_i, scalar_first=True).inv()
                    * r_robot
                ).magnitude()
            )
            errors.append((err_deg, idx, canonicalize_quat(q_bvh_i), canonicalize_quat(q_updated_i)))

        print("=== Scan Result ===")
        print(f"bvh_file   = {args.bvh_file}")
        print(f"robot_file = {args.robot_file}")
        print(f"mapping    = {args.bvh_bone} -> {args.robot_link}")
        print(f"json offset from {table_name} = {np.array2string(canonicalize_quat(q_json), precision=6)}")
        print(f"robot q = {np.array2string(canonicalize_quat(q_robot), precision=6)}")
        print(f"\nTop {min(args.topk, len(errors))} frames by angle(q_bvh * q_json, q_robot):")
        for err_deg, idx, q_bvh_i, q_updated_i in sorted(errors, key=lambda x: x[0])[: args.topk]:
            print(
                f"  frame={idx:5d}  err={err_deg:9.6f} deg  "
                f"q_bvh={np.array2string(q_bvh_i, precision=4)}  "
                f"q_upd={np.array2string(q_updated_i, precision=4)}"
            )
        return

    frame_idx = args.frame if args.frame >= 0 else len(frames) - 1
    bvh_pose = frames[frame_idx]
    bvh_pos, q_bvh = bvh_pose[args.bvh_bone]

    r_bvh = R.from_quat(normalize_quat(q_bvh), scalar_first=True)

    q_calc = (r_bvh.inv() * r_robot).as_quat(scalar_first=True)
    q_updated_json = (r_bvh * r_json).as_quat(scalar_first=True)

    rot_error_json_deg = np.degrees((r_json.inv() * r_bvh.inv() * r_robot).magnitude())
    updated_to_robot_deg = np.degrees(
        (
            R.from_quat(q_updated_json, scalar_first=True).inv()
            * r_robot
        ).magnitude()
    )

    print("=== LAFAN1 BVH -> G1 Link Frame Check ===")
    print(f"bvh_file   = {args.bvh_file}")
    print(f"robot_file = {args.robot_file}")
    print(f"config     = {args.config}")
    print(f"frame      = {frame_idx} / {len(frames) - 1}")
    print(f"mapping    = {args.bvh_bone} -> {args.robot_link}")

    print_frame_axes(f"BVH bone after load_lafan1_file: {args.bvh_bone}", bvh_pos, q_bvh)
    print_frame_axes(f"G1 robot body in neutral pose: {args.robot_link}", robot_pos, q_robot)

    print("\n[Offsets]")
    print(f"computed q_offset = q_bvh^-1 * q_robot:")
    print(f"  {np.array2string(canonicalize_quat(q_calc), precision=6)}")
    print(f"json q_offset from {table_name}:")
    print(f"  {np.array2string(canonicalize_quat(q_json), precision=6)}")
    print(f"json-updated q = q_bvh * q_json:")
    print(f"  {np.array2string(canonicalize_quat(q_updated_json), precision=6)}")
    print(f"robot q:")
    print(f"  {np.array2string(canonicalize_quat(q_robot), precision=6)}")
    print(f"\nangle(json_offset, computed_offset) = {rot_error_json_deg:.6f} deg")
    print(f"angle(q_bvh * q_json, q_robot)      = {updated_to_robot_deg:.6f} deg")


if __name__ == "__main__":
    main()
