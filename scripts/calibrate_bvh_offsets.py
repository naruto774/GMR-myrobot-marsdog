"""Calibrate BVH-to-robot offsets and global scale from a reference frame.

The value written to JSON is GMR's rot_offset, not the final target
orientation. GMR applies it as: updated_quat = bvh_quat_world * rot_offset.

The global scale is estimated from all configured BVH-bone / robot-link
landmarks relative to their roots.  It is written as
``human_height_assumption`` because GMR computes the applied scale as:

    runtime_actual_human_height / human_height_assumption

For dog BVH retargeting, ``runtime_actual_human_height`` is currently the
``--robot_body_length`` passed to scripts/dog_bvh_to_marsdog.py.
"""

import argparse
import json
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from general_motion_retargeting.params import IK_CONFIG_ROOT, ROBOT_XML_DICT
from general_motion_retargeting.utils.dog_bvh import load_dog_bvh_file
from general_motion_retargeting.utils.marsdog_axis import (
    MARSDOG_AXIS_CORRECTION_QUAT_WXYZ,
    apply_marsdog_axis_correction_to_frames,
)

#计算旋转偏移量
def compute_rot_offset(q_bvh, q_robot):
    """GMR convention: updated_quat = q_bvh_world * rot_offset."""
    r_bvh = R.from_quat(q_bvh, scalar_first=True)
    r_robot = R.from_quat(q_robot, scalar_first=True)
    r_off = r_bvh.inv() * r_robot
    return r_off.as_quat(scalar_first=True)

#格式化向量
def fmt_vec(v, decimals=6):
    return [round(float(x), decimals) for x in v]

#格式化四元数
def fmt_quat(q, decimals=6):
    return [round(float(x), decimals) for x in q]

#获取机器人链接位置
def get_robot_link_poses():
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["marsdog"]))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    poses = {}
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name:
            poses[name] = (data.xpos[i].copy(), data.xquat[i].copy())
    return poses


def compute_global_scale(bvh_pose, robot_pose, config):
    """Fit one scale from root-relative matched landmark positions.

    Solve min_s sum_i ||s * (p_bvh_i - p_bvh_root)
                         - (p_robot_i - p_robot_root)||^2.
    """
    bvh_root_name = config["human_root_name"]
    robot_root_name = config["robot_root_name"]
    bvh_root_pos = np.asarray(bvh_pose[bvh_root_name][0], dtype=float)
    robot_root_pos = np.asarray(robot_pose[robot_root_name][0], dtype=float)

    source_vectors = []
    target_vectors = []
    seen_pairs = set()
    for table_name in ("ik_match_table1", "ik_match_table2"):
        for robot_link, entry in config[table_name].items():
            bvh_bone = entry[0]
            pair = (robot_link, bvh_bone)
            if pair in seen_pairs or bvh_bone == bvh_root_name:
                continue
            seen_pairs.add(pair)
            source_vectors.append(
                np.asarray(bvh_pose[bvh_bone][0], dtype=float) - bvh_root_pos
            )
            target_vectors.append(
                np.asarray(robot_pose[robot_link][0], dtype=float) - robot_root_pos
            )

    source = np.asarray(source_vectors)
    target = np.asarray(target_vectors)
    denominator = float(np.sum(source * source))
    if denominator <= np.finfo(float).eps:
        raise ValueError("Cannot estimate global scale: BVH reference landmarks coincide.")

    scale = float(np.sum(source * target) / denominator)
    if scale <= 0.0:
        raise ValueError(f"Estimated non-positive global scale: {scale}")

    residual = target - scale * source
    rms_error = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return scale, rms_error, len(source_vectors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bvh_file", type=str, default="assets/bvh/d2-tpose-clean.bvh")
    parser.add_argument(
        "--config",
        type=str,
        default=str(IK_CONFIG_ROOT / "bvh_to_cat.json"),
    )
    parser.add_argument("--frame", type=int, default=-1, help="BVH frame index (-1 = last)")
    parser.add_argument(
        "--target-body-length",
        type=float,
        default=0.4,
        help=(
            "Value passed as --robot_body_length at retarget runtime, in meters. "
            "The derived scale is written through human_height_assumption."
        ),
    )
    parser.add_argument(
        "--no-global-scale",
        action="store_true",
        help="Do not estimate or write human_height_assumption.",
    )
    parser.add_argument(
        "--zero-pos-offset",
        action="store_true",
        help="Set all pos_offset fields to [0, 0, 0]. By default, only rot_offset is changed.",
    )
    parser.add_argument(
        "--no-marsdog-axis-correction",
        action="store_true",
        help="Disable the Marsdog-specific global axis correction before calibration.",
    )
    args = parser.parse_args()

    frames, _, _ = load_dog_bvh_file(args.bvh_file)
    if not args.no_marsdog_axis_correction:
        frames = apply_marsdog_axis_correction_to_frames(frames)

    frame_idx = args.frame if args.frame >= 0 else len(frames) - 1
    bvh_pose = frames[frame_idx]
    robot_pose = get_robot_link_poses()

    with open(args.config, "r") as f:
        config = json.load(f)

    if not args.no_global_scale:
        global_scale, rms_error, landmark_count = compute_global_scale(
            bvh_pose, robot_pose, config
        )
        config["human_height_assumption"] = round(
            args.target_body_length / global_scale, 6
        )

    for table_name in ("ik_match_table1", "ik_match_table2"):
        for robot_link, entry in config[table_name].items():
            bvh_bone = entry[0]
            _, q_bvh = bvh_pose[bvh_bone]
            _, q_robot = robot_pose[robot_link]

            q_off = compute_rot_offset(q_bvh, q_robot)
            if args.zero_pos_offset:
                entry[3] = fmt_vec(np.zeros(3))
            entry[4] = fmt_quat(q_off)

    with open(args.config, "w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")

    print(f"Calibrated rot_offset from BVH frame {frame_idx} + marsdog qpos=0")
    if not args.no_global_scale:
        print(
            f"Global scale: {global_scale:.6f} "
            f"({landmark_count} landmarks, RMS residual {rms_error:.6f} m)"
        )
        print(
            "Wrote human_height_assumption: "
            f"{config['human_height_assumption']:.6f} m "
            f"(for --robot_body_length {args.target_body_length:.6f} m)"
        )
    if not args.no_marsdog_axis_correction:
        print(f"Applied marsdog_axis_correction_wxyz: {MARSDOG_AXIS_CORRECTION_QUAT_WXYZ}")
    print(f"Updated: {args.config}\n")
    for table_name in ("ik_match_table1",):
        for robot_link, entry in config[table_name].items():
            bvh_bone = entry[0]
            print(f"{robot_link} <- {bvh_bone}")
            print(f"  rot_offset: {entry[4]}")


if __name__ == "__main__":
    main()
