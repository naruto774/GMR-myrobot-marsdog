#!/usr/bin/env python3
"""Refine a GMR rot_offset from printed static-view quaternions.

GMR applies rotations as:
    q_updated = q_bvh * q_offset

Given q_bvh, current q_offset, and q_robot, this script computes:
    q_err = q_updated^-1 * q_robot
    q_offset_new = q_offset * q_err

All quaternions are in wxyz order.
"""

import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize_quat(quat):
    quat = np.asarray(quat, dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero.")
    return quat / norm


def fmt_quat(quat):
    quat = normalize_quat(quat)
    return "[" + ", ".join(f"{x:.6f}" for x in quat) + "]"


def rot_error_deg(q_a, q_b):
    r_a = R.from_quat(normalize_quat(q_a), scalar_first=True)
    r_b = R.from_quat(normalize_quat(q_b), scalar_first=True)
    return np.degrees((r_a.inv() * r_b).magnitude())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh-quat",
        nargs=4,
        type=float,
        required=True,
        help="BVH quaternion in wxyz order.",
    )
    parser.add_argument(
        "--rot-offset",
        nargs=4,
        type=float,
        required=True,
        help="Current rot_offset in wxyz order.",
    )
    parser.add_argument(
        "--robot-quat",
        nargs=4,
        type=float,
        required=True,
        help="Target robot quaternion in wxyz order.",
    )
    args = parser.parse_args()

    q_bvh = normalize_quat(args.bvh_quat)
    q_offset = normalize_quat(args.rot_offset)
    q_robot = normalize_quat(args.robot_quat)

    r_bvh = R.from_quat(q_bvh, scalar_first=True)
    r_offset = R.from_quat(q_offset, scalar_first=True)
    r_robot = R.from_quat(q_robot, scalar_first=True)

    r_updated = r_bvh * r_offset
    q_updated = r_updated.as_quat(scalar_first=True)

    r_err = r_updated.inv() * r_robot
    q_err = r_err.as_quat(scalar_first=True)

    r_offset_new = r_offset * r_err
    q_offset_new = r_offset_new.as_quat(scalar_first=True)
    q_updated_new = (r_bvh * r_offset_new).as_quat(scalar_first=True)

    print("Convention: wxyz, GMR uses q_updated = q_bvh * q_offset")
    print(f"q_bvh          = {fmt_quat(q_bvh)}")
    print(f"q_offset_old   = {fmt_quat(q_offset)}")
    print(f"q_updated_old  = {fmt_quat(q_updated)}")
    print(f"q_robot        = {fmt_quat(q_robot)}")
    print(f"old_error_deg  = {rot_error_deg(q_updated, q_robot):.8f}")
    print()
    print(f"q_err          = updated^-1 * robot = {fmt_quat(q_err)}")
    print(f"q_err_rotvec_deg = {np.array2string(np.degrees(r_err.as_rotvec()), precision=6)}")
    print()
    print(f"q_offset_new   = offset_old * q_err = {fmt_quat(q_offset_new)}")
    print(f"q_updated_new  = {fmt_quat(q_updated_new)}")
    print(f"new_error_deg  = {rot_error_deg(q_updated_new, q_robot):.8f}")


if __name__ == "__main__":
    main()

