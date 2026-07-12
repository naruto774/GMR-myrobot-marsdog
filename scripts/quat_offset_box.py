"""Visualize and verify quaternion right-multiplication on a MuJoCo box.

Example:
    python scripts/quat_offset_box.py \
        --quat 1 0 0 0 \
        --rot-offset 0.645954 -0.761451 0.036006 0.040504 \
        --target-quat 0.645954 -0.761451 0.036006 0.040504 \
        --viewer
"""

import argparse
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R


def parse_quat(values):
    quat = np.asarray(values, dtype=float)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero.")
    return quat / norm


def quat_to_str(quat):
    return "[" + " ".join(f"{x:.6f}" for x in quat) + "]"


def main():
    parser = argparse.ArgumentParser(
        description="Create a MuJoCo box with quat * rot_offset as its orientation."
    )
    parser.add_argument(
        "--pos",
        nargs=3,
        type=float,
        default=[0.0, 0.0, 0.2],
        help="Box position in MuJoCo world coordinates.",
    )
    parser.add_argument(
        "--quat",
        nargs=4,
        type=float,
        default=[1.0, 0.0, 0.0, 0.0],
        help="Input quaternion in wxyz order.",
    )
    parser.add_argument(
        "--rot-offset",
        nargs=4,
        type=float,
        required=True,
        help="Right-multiplied offset quaternion in wxyz order.",
    )
    parser.add_argument(
        "--target-quat",
        nargs=4,
        type=float,
        default=None,
        help="Optional target/motion quaternion in wxyz order for error checking.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open MuJoCo viewer. Without this flag the script only prints results.",
    )
    args = parser.parse_args()

    quat = parse_quat(args.quat)
    rot_offset = parse_quat(args.rot_offset)

    updated_quat = (
        R.from_quat(quat, scalar_first=True)
        * R.from_quat(rot_offset, scalar_first=True)
    ).as_quat(scalar_first=True)
    updated_quat = parse_quat(updated_quat)

    print("Quaternion convention: wxyz")
    print(f"quat        = {quat_to_str(quat)}")
    print(f"rot_offset  = {quat_to_str(rot_offset)}")
    print(f"updated     = quat * rot_offset = {quat_to_str(updated_quat)}")

    if args.target_quat is not None:
        target_quat = parse_quat(args.target_quat)
        error_deg = np.degrees(
            (
                R.from_quat(updated_quat, scalar_first=True).inv()
                * R.from_quat(target_quat, scalar_first=True)
            ).magnitude()
        )
        print(f"target      = {quat_to_str(target_quat)}")
        print(f"rot_error   = {error_deg:.8f} deg")

    pos = np.asarray(args.pos, dtype=float)
    xml = f"""
<mujoco model="quat_offset_box">
  <option timestep="0.01"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
    <rgba haze="0.15 0.25 0.35 1" />
    <global azimuth="-140" elevation="-20" />
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" pos="0 0 0.0" size="0 0 0.05" type="plane" material="groundplane" />
    <body name="box" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{updated_quat[0]} {updated_quat[1]} {updated_quat[2]} {updated_quat[3]}">
      <geom type="box" size="0.1 0.1 0.2" rgba="0.1 0.4 0.9 1"/>
      <site name="box_frame" pos="0 0 0" size="0.02" rgba="1 0 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    print(f"mujoco xpos  = {quat_to_str(data.xpos[box_id])}")
    print(f"mujoco xquat = {quat_to_str(data.xquat[box_id])}")

    if args.viewer:
        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
