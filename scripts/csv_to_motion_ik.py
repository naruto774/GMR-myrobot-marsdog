"""Offline root reconstruction from leg joint CSV via stance-foot kinematics.

Reads ql/qr leg joints, assumes the lower foot is in stance, and back-solves
floating-base pose frame by frame. Output is a GMR-compatible motion pickle.
"""

import argparse
import pickle
from pathlib import Path

import mujoco as mj
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation as R

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "ql_qr_solve_processed.csv"
DEFAULT_XML = ROOT / "assets/myrobot/meshes/myrobot.xml"
DEFAULT_OUT = ROOT / "raw_smpl/zmp_walk_ik.pkl"

LEFT_FOOT = "left_ankle_roll_link"
RIGHT_FOOT = "right_ankle_roll_link"
FOOT_GEOM_NAMES = ("l_foot_1", "l_foot_2", "r_foot_1", "r_foot_2")
NUM_LEG_DOF = 12


def get_foot_geom_ids(model: mj.MjModel) -> list[int]:
    return [model.geom(name).id for name in FOOT_GEOM_NAMES]


def geom_lowest_world_z(model: mj.MjModel, data: mj.MjData, geom_id: int) -> float:
    """Lowest world-frame z of a MuJoCo geom (conservative for cylinders)."""
    pos = data.geom_xpos[geom_id]
    if model.geom_type[geom_id] == mj.mjtGeom.mjGEOM_CYLINDER:
        mat = data.geom_xmat[geom_id].reshape(3, 3)
        half = model.geom_size[geom_id][1]
        radius = model.geom_size[geom_id][0]
        axis = mat[:, 2]
        cap_z = [
            (pos + axis * local_z)[2]
            for local_z in (-half, half)
        ]
        return min(cap_z) - radius
    return pos[2]


def foot_sole_lowest_z(model: mj.MjModel, data: mj.MjData, foot_geom_ids: list[int]) -> float:
    return min(geom_lowest_world_z(model, data, gid) for gid in foot_geom_ids)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline IK: CSV leg joints -> GMR motion pkl with root pose",
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--root-z-hint",
        type=float,
        default=0.23,
        help="Base height used only for stance-foot detection",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=0.0,
        help="Resample to this fps (0 = keep original)",
    )
    parser.add_argument(
        "--stance-filter",
        type=int,
        default=11,
        help="Median filter window for stance-foot detection (odd, >= 3)",
    )
    parser.add_argument(
        "--min-stance-frames",
        type=int,
        default=8,
        help="Minimum frames before allowing a stance switch",
    )
    parser.add_argument(
        "--smooth-root",
        type=int,
        default=7,
        help="Savitzky-Golay window for root smoothing (0 = disable, odd)",
    )
    return parser.parse_args()


def load_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    t = df["time_perf"].values.astype(np.float64)
    ql = df[[f"ql_{i}" for i in range(6)]].values.astype(np.float64)
    qr = df[[f"qr_{i}" for i in range(6)]].values.astype(np.float64)
    return t, ql, qr


def resample_motion(t, ql, qr, target_fps: float):
    if target_fps <= 0:
        dt = float(np.median(np.diff(t)))
        return t, ql, qr, 1.0 / dt

    t0, t1 = t[0], t[-1]
    dt = 1.0 / target_fps
    t_new = np.arange(t0, t1, dt)
    if t_new[-1] < t1:
        t_new = np.append(t_new, t1)

    ql_new = np.empty((len(t_new), 6), dtype=np.float64)
    qr_new = np.empty((len(t_new), 6), dtype=np.float64)
    for j in range(6):
        ql_new[:, j] = np.interp(t_new, t, ql[:, j])
        qr_new[:, j] = np.interp(t_new, t, qr[:, j])
    return t_new, ql_new, qr_new, target_fps


def assemble_leg_dof(qr_row: np.ndarray, ql_row: np.ndarray) -> np.ndarray:
    leg = np.zeros(NUM_LEG_DOF, dtype=np.float64)
    leg[0:6] = qr_row
    leg[6:12] = ql_row
    return leg


def make_homogeneous(rot: np.ndarray, pos: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    transform[:3, 3] = pos
    return transform


def mat_to_wxyz(rot: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_matrix(rot).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])


def set_leg_qpos(model: mj.MjModel, data: mj.MjData, leg_dof: np.ndarray):
    data.qpos[7:7 + NUM_LEG_DOF] = leg_dof
    if model.nq > 7 + NUM_LEG_DOF:
        data.qpos[7 + NUM_LEG_DOF :] = 0.0


def foot_pose_in_base(
    model: mj.MjModel,
    data: mj.MjData,
    leg_dof: np.ndarray,
    foot_body_id: int,
) -> np.ndarray:
    data.qpos[:3] = 0.0
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    set_leg_qpos(model, data, leg_dof)
    mj.mj_forward(model, data)
    foot_rot = data.xmat[foot_body_id].reshape(3, 3).copy()
    foot_pos = data.xpos[foot_body_id].copy()
    return make_homogeneous(foot_rot, foot_pos)


def foot_sole_heights(
    model: mj.MjModel,
    data: mj.MjData,
    leg_dof: np.ndarray,
    root_z: float,
    left_geom_ids: list[int],
    right_geom_ids: list[int],
) -> tuple[float, float]:
    data.qpos[:3] = np.array([0.0, 0.0, root_z])
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    set_leg_qpos(model, data, leg_dof)
    mj.mj_forward(model, data)
    z_left = foot_sole_lowest_z(model, data, left_geom_ids)
    z_right = foot_sole_lowest_z(model, data, right_geom_ids)
    return z_left, z_right


def detect_stance(
    model: mj.MjModel,
    data: mj.MjData,
    ql: np.ndarray,
    qr: np.ndarray,
    root_z_hint: float,
    filter_window: int,
    min_stance_frames: int,
) -> np.ndarray:
    left_geom_ids = [model.geom("l_foot_1").id, model.geom("l_foot_2").id]
    right_geom_ids = [model.geom("r_foot_1").id, model.geom("r_foot_2").id]
    num_frames = len(ql)
    raw = np.zeros(num_frames, dtype=np.int32)

    for t in range(num_frames):
        leg = assemble_leg_dof(qr[t], ql[t])
        z_left, z_right = foot_sole_heights(
            model, data, leg, root_z_hint, left_geom_ids, right_geom_ids,
        )
        raw[t] = 0 if z_left <= z_right else 1

    window = max(3, filter_window | 1)
    smoothed = median_filter(raw.astype(np.float64), size=window, mode="nearest")
    smoothed = (smoothed >= 0.5).astype(np.int32)

    debounced = smoothed.copy()
    current = debounced[0]
    hold = 1
    for t in range(1, num_frames):
        if smoothed[t] == current:
            hold += 1
            debounced[t] = current
            continue
        if hold < min_stance_frames:
            debounced[t] = current
            hold += 1
        else:
            current = smoothed[t]
            debounced[t] = current
            hold = 1

    left_ratio = np.mean(debounced == 0)
    print(f"stance detection: left={left_ratio:.1%}, right={1-left_ratio:.1%}")
    return debounced


def reconstruct_root(
    model: mj.MjModel,
    data: mj.MjData,
    ql: np.ndarray,
    qr: np.ndarray,
    stance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    left_id = model.body(LEFT_FOOT).id
    right_id = model.body(RIGHT_FOOT).id
    num_frames = len(ql)
    root_pos = np.zeros((num_frames, 3), dtype=np.float64)
    root_rot_wxyz = np.zeros((num_frames, 4), dtype=np.float64)

    current_stance = int(stance[0])
    foot_anchor = np.eye(4, dtype=np.float64)

    for t in range(num_frames):
        leg = assemble_leg_dof(qr[t], ql[t])
        foot_id = left_id if current_stance == 0 else right_id
        base_to_foot = foot_pose_in_base(model, data, leg, foot_id)
        world_to_base = foot_anchor @ np.linalg.inv(base_to_foot)

        root_pos[t] = world_to_base[:3, 3]
        root_rot_wxyz[t] = mat_to_wxyz(world_to_base[:3, :3])

        if t + 1 < num_frames and stance[t + 1] != current_stance:
            new_stance = int(stance[t + 1])
            new_foot_id = left_id if new_stance == 0 else right_id
            new_base_to_foot = foot_pose_in_base(model, data, leg, new_foot_id)
            foot_anchor = world_to_base @ new_base_to_foot
            current_stance = new_stance

    return root_pos, root_rot_wxyz


def smooth_root_pos(root_pos: np.ndarray, window: int) -> np.ndarray:
    if window <= 2:
        return root_pos
    window = window | 1
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        return root_pos
    smoothed = root_pos.copy()
    for axis in range(3):
        smoothed[:, axis] = savgol_filter(
            root_pos[:, axis],
            window_length=min(window, len(root_pos) // 2 * 2 - 1),
            polyorder=2,
            mode="interp",
        )
    return smoothed


def align_to_ground(
    model: mj.MjModel,
    data: mj.MjData,
    root_pos: np.ndarray,
    root_rot_wxyz: np.ndarray,
    dof_pos: np.ndarray,
    foot_geom_ids: list[int],
    ground_clearance: float = 0.0,
) -> np.ndarray:
    lowest = np.inf
    for t in range(len(root_pos)):
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = root_rot_wxyz[t]
        data.qpos[7:] = dof_pos[t]
        mj.mj_forward(model, data)
        lowest = min(lowest, foot_sole_lowest_z(model, data, foot_geom_ids))

    offset = ground_clearance - lowest
    root_pos = root_pos.copy()
    root_pos[:, 2] += offset
    print(
        f"ground align (foot sole): shifted root z by {offset:.4f} m "
        f"(clearance={ground_clearance:.4f} m)"
    )
    return root_pos


def report_quality(
    model: mj.MjModel,
    data: mj.MjData,
    root_pos: np.ndarray,
    root_rot_wxyz: np.ndarray,
    dof_pos: np.ndarray,
    left_geom_ids: list[int],
    right_geom_ids: list[int],
):
    num_frames = len(root_pos)
    foot_z_min = []
    foot_z_max = []
    root_jerk = []

    for t in range(num_frames):
        data.qpos[:3] = root_pos[t]
        data.qpos[3:7] = root_rot_wxyz[t]
        data.qpos[7:] = dof_pos[t]
        mj.mj_forward(model, data)
        z_l = foot_sole_lowest_z(model, data, left_geom_ids)
        z_r = foot_sole_lowest_z(model, data, right_geom_ids)
        foot_z_min.append(min(z_l, z_r))
        foot_z_max.append(max(z_l, z_r))

    root_vel = np.diff(root_pos, axis=0)
    root_acc = np.diff(root_vel, axis=0)
    root_jerk = np.linalg.norm(np.diff(root_acc, axis=0), axis=1)

    print("quality metrics (foot sole geoms):")
    print(f"  sole z (lower foot): mean={np.mean(foot_z_min):.4f}, max={np.max(foot_z_min):.4f}")
    print(f"  sole z (higher foot): mean={np.mean(foot_z_max):.4f}, max={np.max(foot_z_max):.4f}")
    print(f"  root travel xy: {np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2]):.4f} m")
    print(f"  root height range: [{root_pos[:, 2].min():.4f}, {root_pos[:, 2].max():.4f}] m")
    if len(root_jerk) > 0:
        print(f"  root jerk: mean={np.mean(root_jerk):.4f}, max={np.max(root_jerk):.4f}")


def main():
    args = parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not args.xml.is_file():
        raise FileNotFoundError(f"MuJoCo XML not found: {args.xml}")

    t, ql, qr = load_csv(args.csv)
    t, ql, qr, fps = resample_motion(t, ql, qr, args.target_fps)
    print(f"csv={args.csv}")
    print(f"frames={len(t)}, duration={t[-1]-t[0]:.2f}s, fps={fps:.2f}")

    model = mj.MjModel.from_xml_path(str(args.xml))
    data = mj.MjData(model)
    num_full_dof = model.nq - 7
    print(f"xml={args.xml}, actuated dof={num_full_dof}")

    stance = detect_stance(
        model,
        data,
        ql,
        qr,
        args.root_z_hint,
        args.stance_filter,
        args.min_stance_frames,
    )
    root_pos, root_rot_wxyz = reconstruct_root(model, data, ql, qr, stance)

    if args.smooth_root > 2:
        root_pos = smooth_root_pos(root_pos, args.smooth_root)

    dof_pos = np.zeros((len(t), num_full_dof), dtype=np.float64)
    for i in range(len(t)):
        dof_pos[i, :NUM_LEG_DOF] = assemble_leg_dof(qr[i], ql[i])

    foot_geom_ids = get_foot_geom_ids(model)
    left_geom_ids = foot_geom_ids[:2]
    right_geom_ids = foot_geom_ids[2:]
    root_pos = align_to_ground(
        model, data, root_pos, root_rot_wxyz, dof_pos, foot_geom_ids,
    )

    root_pos[:, 0] -= root_pos[0, 0]
    root_pos[:, 1] -= root_pos[0, 1]

    root_rot_xyzw = np.array([wxyz_to_xyzw(q) for q in root_rot_wxyz])

    report_quality(
        model, data, root_pos, root_rot_wxyz, dof_pos, left_geom_ids, right_geom_ids,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    motion_data = {
        "fps": fps,
        "root_pos": root_pos,
        "root_rot": root_rot_xyzw,
        "dof_pos": dof_pos,
        "local_body_pos": None,
        "link_body_list": None,
    }
    with open(args.output, "wb") as f:
        pickle.dump(motion_data, f)

    print(f"saved: {args.output}")
    print("visualize with:")
    print(f"  python scripts/vis_robot_motion.py --robot myrobot --robot_motion_path {args.output}")


if __name__ == "__main__":
    main()
