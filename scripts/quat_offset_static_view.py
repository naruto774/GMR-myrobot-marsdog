"""Static MuJoCo viewer for manual rot_offset tuning.

Edit ROT_OFFSETS and ACTIVE_ROBOT_LINK at the top of this file, then rerun:

    python scripts/quat_offset_static_view.py

The viewer shows, for the active mapping:
  - axis-corrected BVH bone frame (small, at BVH position shifted near robot Hips)
  - offset BVH frame              (medium, at robot link position)
  - robot link frame              (large, at robot link position)

If rot_offset is correct, the medium and large frames should overlap.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco as mj
import numpy as np
from mujoco import viewer
from scipy.spatial.transform import Rotation as R

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from general_motion_retargeting.params import IK_CONFIG_ROOT, ROBOT_XML_DICT
from general_motion_retargeting.robot_motion_viewer import draw_frame
from general_motion_retargeting.utils.dog_bvh import _resolve_root_name, load_dog_bvh_file
from general_motion_retargeting.utils.marsdog_axis import (
    MARSDOG_AXIS_CORRECTION_QUAT_WXYZ,
    apply_marsdog_axis_correction_to_frames,
)


BVH_FILE = "assets/bvh/1121-clean.bvh"
CONFIG_PATH = IK_CONFIG_ROOT / "bvh_to_dison.json"
IK_TABLE_NAME = "ik_match_table1"
FRAME_IDX = 0

ACTIVE_ROBOT_LINK = "waist_yaw_link"   # 或 "base_link" 先看躯干

# Refine the active link once and write the new rot_offset back to JSON.
# The viewer will then show the refined result.
REFINE_ACTIVE_ROT_OFFSET = True
WRITE_REFINED_OFFSET_TO_JSON = True

# 手动填写各 BVH bone 的 rot_offset，格式 wxyz
# GMR 约定: updated_quat = bvh_quat_world * rot_offset
ROT_OFFSETS = {
    # "Hips": [0.485398, -0.563887, -0.512967, -0.42811],
    # "NewBone_2":             [
    #             0.542118,
    #             -0.455461,
    #             -0.453223,
    #             -0.541528
    #         ],
    # "NewBone_7":             [
    #             0.594122,
    #             -0.344608,
    #             -0.306325,
    #             -0.659112
    #         ],
    # "NewBone_12":             [
    #             0.662146,
    #             -0.177273,
    #             -0.126891,
    #             -0.716963
    #         ],
    # "NewBone_21":             [
    #             0.484848,
    #             -0.48485,
    #             -0.468459,
    #             -0.557125
    #         ],
    # "NewBone_3":             [
    #             0.370374,
    #             -0.588531,
    #             -0.569669,
    #             -0.4381
    #         ],
    # "NewBone_4":             [
    #             0.564029,
    #             -0.406715,
    #             -0.363834,
    #             -0.619742
    #         ],
    # "NewBone_8":             [
    #             0.462318,
    #             -0.533953,
    #             -0.614997,
    #             -0.35062
    #         ],
    # "NewBone_9":             [
    #             0.555906,
    #             -0.435675,
    #             -0.536585,
    #             -0.46177
    #         ],
    # "back3": [0.707107, -0.000483, 0.000483, -0.707107],
    # "NewBone_13": [0.5, -0.5, -0.5, -0.5],
    # "NewBone_16": [0.5, -0.5, -0.5, -0.5],
    # "Should_R": [0.5, -0.5, -0.5, -0.5],
    # "NewBone_17": [0.5, -0.5, -0.5, -0.5],
    # "NewBone_20": [0.5, -0.5, -0.5, -0.5],
}

FRAME_SIZE_RAW_BVH = 0.80
FRAME_SIZE_UPDATED = 1.00
FRAME_SIZE_ROBOT = 0.40
FRAME_SIZE_ALL_UPDATED = 0.28
FRAME_SIZE_ALL_ROBOT = 0.18

# Draw every mapped robot/updated frame in addition to the active one.
SHOW_ALL_MAPPINGS = True
REFINE_ACTIVE_ROT_OFFSET = False
WRITE_REFINED_OFFSET_TO_JSON = False

def load_ik_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def load_ik_mappings(config, table_name):
    return {
        robot_link: (entry[0], entry[4])
        for robot_link, entry in config[table_name].items()
    }


def normalize_quat(quat):
    quat = np.asarray(quat, dtype=float)
    return quat / np.linalg.norm(quat)


def apply_rot_offset(q_bvh, q_offset):
    return (
        R.from_quat(normalize_quat(q_bvh), scalar_first=True)
        * R.from_quat(normalize_quat(q_offset), scalar_first=True)
    ).as_quat(scalar_first=True)


def refine_rot_offset(q_bvh, q_offset, q_robot):
    """One-step correction: q_new = q_offset * ((q_bvh*q_offset)^-1*q_robot)."""
    r_bvh = R.from_quat(normalize_quat(q_bvh), scalar_first=True)
    r_offset = R.from_quat(normalize_quat(q_offset), scalar_first=True)
    r_robot = R.from_quat(normalize_quat(q_robot), scalar_first=True)

    r_updated = r_bvh * r_offset
    r_err = r_updated.inv() * r_robot
    r_offset_new = r_offset * r_err
    return (
        r_err.as_quat(scalar_first=True),
        r_offset_new.as_quat(scalar_first=True),
        np.degrees(r_err.as_rotvec()),
    )


def rot_error_deg(q_a, q_b):
    return np.degrees(
        (
            R.from_quat(normalize_quat(q_a), scalar_first=True).inv()
            * R.from_quat(normalize_quat(q_b), scalar_first=True)
        ).magnitude()
    )


def bvh_vis_position(pos, root_pos):
    """Shift BVH skeleton so root sits at world origin for easier comparison."""
    return np.asarray(pos, dtype=float) - np.asarray(root_pos, dtype=float)


def quat_str(quat):
    return np.array2string(np.asarray(quat, dtype=float), precision=6)


def fmt_quat_list(quat, decimals=6):
    quat = normalize_quat(quat)
    return [round(float(x), decimals) for x in quat]


def write_active_offset_to_json(config_path, robot_link, q_offset_new):
    config = load_ik_config(config_path)
    updated_tables = []
    for table_name in ("ik_match_table1", "ik_match_table2"):
        if robot_link in config[table_name]:
            config[table_name][robot_link][4] = fmt_quat_list(q_offset_new)
            updated_tables.append(table_name)

    if not updated_tables:
        raise KeyError(f"Cannot write offset: {robot_link} not found in IK tables.")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
        f.write("\n")

    return updated_tables


def print_pair_report(robot_link, bvh_bone, q_bvh, q_offset, q_robot):
    q_updated = apply_rot_offset(q_bvh, q_offset)
    print(f"\nActive mapping: {robot_link} <- {bvh_bone}")
    print(f"  bvh quat      = {quat_str(q_bvh)}")
    print(f"  rot_offset    = {quat_str(q_offset)}")
    print(f"  updated quat  = {quat_str(q_updated)}")
    print(f"  robot quat    = {quat_str(q_robot)}")
    print(f"  rot_error     = {rot_error_deg(q_updated, q_robot):.6f} deg")


def build_mapping_visuals(mappings, bvh_pose, data):
    visuals = []
    for robot_link, (bvh_bone, config_offset) in mappings.items():
        if bvh_bone not in bvh_pose:
            print(f"[warn] BVH bone not found, skip: {robot_link} <- {bvh_bone}")
            continue

        link_id = mj.mj_name2id(data.model, mj.mjtObj.mjOBJ_BODY, robot_link)
        if link_id < 0:
            print(f"[warn] Robot link not found, skip: {robot_link}")
            continue

        _bvh_pos, q_bvh = bvh_pose[bvh_bone]
        q_offset = ROT_OFFSETS.get(bvh_bone, config_offset)
        q_updated = apply_rot_offset(q_bvh, q_offset)
        q_robot = data.xquat[link_id].copy()

        visuals.append(
            {
                "robot_link": robot_link,
                "bvh_bone": bvh_bone,
                "link_pos": data.xpos[link_id].copy(),
                "link_mat": data.xmat[link_id].reshape(3, 3).copy(),
                "updated_mat": R.from_quat(q_updated, scalar_first=True).as_matrix(),
                "rot_error": rot_error_deg(q_updated, q_robot),
            }
        )
    return visuals


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize BVH rot_offset alignment against Marsdog link frames."
    )
    parser.add_argument("--bvh_file", type=str, default=BVH_FILE)
    parser.add_argument("--config", type=str, default=str(CONFIG_PATH))
    parser.add_argument("--active_link", type=str, default=ACTIVE_ROBOT_LINK)
    parser.add_argument("--frame", type=int, default=FRAME_IDX)
    parser.add_argument(
        "--table",
        choices=("ik_match_table1", "ik_match_table2"),
        default=IK_TABLE_NAME,
        help="IK mapping table to inspect.",
    )
    parser.add_argument(
        "--no-marsdog-axis-correction",
        action="store_true",
        help="Disable Marsdog global axis correction for debugging.",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        default=SHOW_ALL_MAPPINGS,
        help="Draw all mapped updated/robot frames and print all rot errors.",
    )
    parser.add_argument(
        "--refine-active",
        action="store_true",
        help="Compute a one-step corrected rot_offset for the active link.",
    )
    parser.add_argument(
        "--write-refined",
        action="store_true",
        help="Write the refined active rot_offset back to the JSON config.",
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
    config_path = resolve_path(args.config)

    config = load_ik_config(config_path)
    mappings = load_ik_mappings(config, args.table)
    if args.active_link not in mappings:
        raise KeyError(f"Unknown ACTIVE_ROBOT_LINK in {args.table}: {args.active_link}")

    bvh_bone, config_offset = mappings[args.active_link]

    frames, _, _ = load_dog_bvh_file(str(bvh_file))
    if not args.no_marsdog_axis_correction:
        frames = apply_marsdog_axis_correction_to_frames(frames)
    frame_idx = args.frame if args.frame >= 0 else len(frames) - 1
    bvh_pose = frames[frame_idx]
    root_name = _resolve_root_name(frames)
    root_pos, _ = bvh_pose[root_name]
    bvh_pos, q_bvh = bvh_pose[bvh_bone]
    q_offset = ROT_OFFSETS.get(bvh_bone, config_offset)
    q_updated = apply_rot_offset(q_bvh, q_offset)

    model = mj.MjModel.from_xml_path(str(ROBOT_XML_DICT["marsdog"]))
    data = mj.MjData(model)
    mj.mj_forward(model, data)

    link_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, args.active_link)
    if link_id < 0:
        raise ValueError(f"Robot link not found in model: {args.active_link}")

    link_pos = data.xpos[link_id].copy()
    link_mat = data.xmat[link_id].reshape(3, 3).copy()
    q_robot = data.xquat[link_id].copy()

    print(f"\nbvh_file      = {bvh_file}")
    print(f"config        = {config_path}")
    print(f"table         = {args.table}")
    print(f"frame         = {frame_idx} / {len(frames) - 1}")
    print(f"marsdog_axis  = {not args.no_marsdog_axis_correction}")
    print("\n=== Before refinement ===")
    print_pair_report(args.active_link, bvh_bone, q_bvh, q_offset, q_robot)
    if args.refine_active:
        q_err, q_offset_new, q_err_rotvec_deg = refine_rot_offset(
            q_bvh, q_offset, q_robot
        )
        q_updated_new = apply_rot_offset(q_bvh, q_offset_new)
        print("\n=== Refinement ===")
        print(f"  q_err          = {quat_str(q_err)}")
        print(f"  q_err_rotvec   = {np.array2string(q_err_rotvec_deg, precision=6)} deg")
        print(f"  rot_offset_new = {quat_str(q_offset_new)}")
        print(f"  updated_new    = {quat_str(q_updated_new)}")
        print(f"  new_rot_error  = {rot_error_deg(q_updated_new, q_robot):.6f} deg")

        q_offset = q_offset_new
        q_updated = q_updated_new

        if args.write_refined:
            updated_tables = write_active_offset_to_json(config_path, args.active_link, q_offset_new)
            print(f"  wrote JSON     = {config_path}")
            print(f"  updated tables = {updated_tables}")

    print(f"marsdog_axis_correction_wxyz: {quat_str(MARSDOG_AXIS_CORRECTION_QUAT_WXYZ)}")
    print("\nViewer legend:")
    print("  small frame  : axis-corrected BVH bone orientation")
    print("  medium frame : BVH quat * rot_offset")
    print("  large frame  : robot link orientation")
    print("Close the viewer window and edit ROT_OFFSETS, then rerun.")

    raw_pos = bvh_vis_position(bvh_pos, root_pos)
    raw_mat = R.from_quat(q_bvh, scalar_first=True).as_matrix()
    updated_mat = R.from_quat(q_updated, scalar_first=True).as_matrix()
    all_visuals = build_mapping_visuals(mappings, bvh_pose, data)

    if args.show_all:
        print("\nAll mapping rot_error summary:")
        for item in sorted(all_visuals, key=lambda x: x["rot_error"], reverse=True):
            marker = "  *" if item["robot_link"] == args.active_link else "   "
            print(
                f"{marker} {item['robot_link']} <- {item['bvh_bone']}: "
                f"{item['rot_error']:.6f} deg"
            )

    with viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as v:
        v.cam.lookat[:] = link_pos
        v.cam.distance = 1.2
        v.cam.elevation = -15
        v.cam.azimuth = 120

        while v.is_running():
            mj.mj_forward(model, data)
            v.user_scn.ngeom = 0

            if args.show_all:
                for item in all_visuals:
                    if item["robot_link"] == args.active_link:
                        continue
                    draw_frame(
                        item["link_pos"],
                        item["updated_mat"],
                        v,
                        FRAME_SIZE_ALL_UPDATED,
                        joint_name=f"u:{item['robot_link']}",
                    )
                    draw_frame(
                        item["link_pos"],
                        item["link_mat"],
                        v,
                        FRAME_SIZE_ALL_ROBOT,
                        joint_name=f"r:{item['robot_link']}",
                    )

            draw_frame(
                raw_pos,
                raw_mat,
                v,
                FRAME_SIZE_RAW_BVH,
                joint_name=f"raw:{bvh_bone}",
            )
            draw_frame(
                link_pos,
                updated_mat,
                v,
                FRAME_SIZE_UPDATED,
                joint_name=f"updated:{bvh_bone}",
            )
            draw_frame(
                link_pos,
                link_mat,
                v,
                FRAME_SIZE_ROBOT,
                joint_name=f"robot:{args.active_link}",
            )

            v.sync()
            time.sleep(0.02)


if __name__ == "__main__":
    main()
