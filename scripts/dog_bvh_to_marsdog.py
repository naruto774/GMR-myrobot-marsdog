import argparse
import os
import pickle
import time

import numpy as np
from rich import print
from tqdm import tqdm

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.dog_bvh import load_dog_bvh_file
from general_motion_retargeting.utils.marsdog_axis import (
    MARSDOG_AXIS_CORRECTION_QUAT_WXYZ,
    apply_marsdog_axis_correction_to_frames,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bvh_file",
        help="Dog BVH motion file to load.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--robot",
        choices=["marsdog", "unitree_go1", "unitree_go2"],
        default="marsdog",
    )
    parser.add_argument(
        "--robot_body_length",
        type=float,
        default=0.4,
        help="Target marsdog body length in meters (BVH reference is 0.70 m).",
    )
    parser.add_argument(
        "--motion_fps",
        type=int,
        default=None,
        help="Playback/save FPS. Defaults to the FPS parsed from the BVH file.",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/marsdog_example.mp4",
    )
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--save_path",
        default=None,
        help="Path to save the robot motion pickle.",
    )

    args = parser.parse_args()

    if args.save_path is not None:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        qpos_list = []

    dog_frames, actual_body_length, bvh_fps = load_dog_bvh_file(
        args.bvh_file,
        robot_body_length=args.robot_body_length,
    )
    
    dog_frames = apply_marsdog_axis_correction_to_frames(dog_frames)

    motion_fps = args.motion_fps if args.motion_fps is not None else bvh_fps

    retargeter = GMR(
        src_human="dog_bvh",
        tgt_robot=args.robot,
        actual_human_height=actual_body_length,
        use_velocity_limit=False, 
    )

    robot_motion_viewer = RobotMotionViewer(
        robot_type=args.robot,
        motion_fps=motion_fps,
        transparent_robot=0,
        record_video=args.record_video,
        video_path=args.video_path,
    )

    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0

    print(f"mocap_frame_rate: {motion_fps}")
    print(f"body_length_scale: {actual_body_length} / 0.70")
    if args.robot == "marsdog":
        print(f"marsdog_axis_correction_wxyz: {MARSDOG_AXIS_CORRECTION_QUAT_WXYZ}")

    pbar = tqdm(total=len(dog_frames), desc="Retargeting")

    for i, dog_data in enumerate(dog_frames):
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time

        pbar.update(1)

        qpos = retargeter.retarget(dog_data)
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            human_motion_data=retargeter.scaled_human_data,
            rate_limit=args.rate_limit,
        )

        if args.save_path is not None:
            qpos_list.append(qpos)

    if args.save_path is not None:
        root_pos = np.array([qpos[:3] for qpos in qpos_list])
        root_rot = np.array([qpos[3:7][[1, 2, 3, 0]] for qpos in qpos_list])
        dof_pos = np.array([qpos[7:] for qpos in qpos_list])
        motion_data = {
            "fps": motion_fps,
            "root_pos": root_pos,
            "root_rot": root_rot,
            "dof_pos": dof_pos,
            "local_body_pos": None,
            "link_body_list": None,
        }
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"Saved to {args.save_path}")

    pbar.close()
    robot_motion_viewer.close()
