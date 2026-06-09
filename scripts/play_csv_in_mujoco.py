"""Play leg joint angles from CSV in MuJoCo (fixed root, in-place gait check)."""

import argparse
import time
from pathlib import Path

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "ql_qr_solve_processed.csv"
DEFAULT_XML = ROOT / "assets/myrobot/meshes/myrobot.xml"


def parse_args():
    parser = argparse.ArgumentParser(description="Play CSV leg joints in MuJoCo")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--root-z", type=float, default=0.23)
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--loop", action="store_true", default=True)
    parser.add_argument("--no-loop", action="store_false", dest="loop")
    return parser.parse_args()


def load_motion(csv_path: Path):
    df = pd.read_csv(csv_path)
    t = df["time_perf"].values
    ql = df[[f"ql_{i}" for i in range(6)]].values
    qr = df[[f"qr_{i}" for i in range(6)]].values
    dt = float(np.median(np.diff(t)))
    return t, ql, qr, dt


def main():
    args = parse_args()
    if not args.csv.is_file():
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    if not args.xml.is_file():
        raise FileNotFoundError(f"MuJoCo XML not found: {args.xml}")

    t, ql, qr, dt = load_motion(args.csv)
    fps = 1.0 / dt
    print(f"csv={args.csv}")
    print(f"frames={len(t)}, duration={t[-1] - t[0]:.2f}s, fps≈{fps:.1f}")

    model = mj.MjModel.from_xml_path(str(args.xml))
    data = mj.MjData(model)
    n_dof = model.nq - 7
    print(f"xml={args.xml}")
    print(f"model nq={model.nq}, actuated dof={n_dof}")

    def set_qpos(frame_idx: int):
        data.qpos[:3] = [0.0, 0.0, args.root_z]
        data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        leg = np.zeros(12, dtype=np.float64)
        leg[0:6] = qr[frame_idx]
        leg[6:12] = ql[frame_idx]

        data.qpos[7:19] = leg
        if model.nq > 19:
            data.qpos[19:] = 0.0

        mj.mj_forward(model, data)

    set_qpos(0)
    for name in ["left_ankle_roll_link", "right_ankle_roll_link"]:
        bid = model.body(name).id
        print(f"{name} z = {data.xpos[bid][2]:.4f}")

    target_dt = dt / args.speed
    with mjv.launch_passive(model, data) as viewer:
        i = 0
        last_wall = time.time()
        while viewer.is_running():
            set_qpos(i)

            now = time.time()
            if now - last_wall >= target_dt:
                if i + 1 < len(t):
                    i += 1
                elif args.loop:
                    i = 0
                last_wall = now

            viewer.sync()


if __name__ == "__main__":
    main()
