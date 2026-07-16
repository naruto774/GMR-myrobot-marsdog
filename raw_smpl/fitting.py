"""Fit four neck/head joint trajectories with a Fourier series.

The default input file is an Excel workbook saved with a .csv suffix:
    raw_smpl/拟合用/head+neck.csv

For each column y(t), the fitted model is:
    y(t) = a0 + sum_k [a_k cos(2*pi*k*t/T) + b_k sin(2*pi*k*t/T)]

where t is time in seconds and T is the sampled motion duration.

Outputs:
    - fitted_head_neck.csv: original columns, fitted columns, and residuals
    - fourier_coefficients.json: coefficients and error metrics
    - fourier_fit.png: plot, if matplotlib is installed
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import numpy as np


DEFAULT_INPUT = Path(__file__).resolve().parent / "拟合用" / "head+neck.csv"
DEFAULT_JOINT_NAMES = [
    "neck_pitch_joint",
    "head_roll_joint",
    "head_yaw_joint",
    "head_pitch_joint",
]
FOURIER_ORDER = 7
DEFAULT_FPS = 30.0
XLSX_MAGIC = b"PK\x03\x04"
XLSX_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def column_index(cell_ref: str) -> int:
    """Convert an Excel cell reference like 'AB12' to a zero-based column index."""
    letters = re.match(r"[A-Z]+", cell_ref.upper())
    if letters is None:
        raise ValueError(f"Invalid Excel cell reference: {cell_ref}")

    idx = 0
    for char in letters.group(0):
        idx = idx * 26 + (ord(char) - ord("A") + 1)
    return idx - 1


def parse_numeric(value: object) -> float:
    if value is None or value == "":
        return math.nan
    return float(value)


def read_xlsx_like(path: Path) -> np.ndarray:
    """Read numeric data from an .xlsx file without requiring openpyxl."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        sheet_names = [
            name
            for name in names
            if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
        ]
        if not sheet_names:
            raise ValueError(f"No worksheet found in {path}")

        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in names:
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("x:si", XLSX_NS):
                text = "".join(node.text or "" for node in item.findall(".//x:t", XLSX_NS))
                shared_strings.append(text)

        sheet = ET.fromstring(archive.read(sheet_names[0]))
        rows: list[list[float]] = []

        for row in sheet.findall(".//x:sheetData/x:row", XLSX_NS):
            cells: dict[int, float] = {}
            for cell in row.findall("x:c", XLSX_NS):
                ref = cell.attrib.get("r")
                if ref is None:
                    continue

                value_node = cell.find("x:v", XLSX_NS)
                if value_node is None:
                    continue

                value: object = value_node.text
                if cell.attrib.get("t") == "s":
                    value = shared_strings[int(value_node.text)]

                cells[column_index(ref)] = parse_numeric(value)

            if cells:
                max_col = max(cells)
                rows.append([cells.get(i, math.nan) for i in range(max_col + 1)])

    return rectangular_array(rows)


def read_text_csv(path: Path) -> np.ndarray:
    """Read a numeric CSV file, allowing an optional non-numeric header row."""
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row:
                continue
            try:
                rows.append([parse_numeric(value.strip()) for value in row])
            except ValueError:
                # Treat a non-numeric row as a header and skip it.
                if rows:
                    raise

    return rectangular_array(rows)


def rectangular_array(rows: Iterable[list[float]]) -> np.ndarray:
    rows = list(rows)
    if not rows:
        raise ValueError("Input file contains no numeric rows.")

    width = max(len(row) for row in rows)
    data = np.full((len(rows), width), math.nan, dtype=np.float64)
    for row_idx, row in enumerate(rows):
        data[row_idx, : len(row)] = row

    valid_columns = ~np.all(np.isnan(data), axis=0)
    data = data[:, valid_columns]
    valid_rows = ~np.any(np.isnan(data), axis=1)
    return data[valid_rows]


def load_motion_table(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("rb") as file:
        magic = file.read(4)

    if magic == XLSX_MAGIC:
        return read_xlsx_like(path)
    return read_text_csv(path)


def sample_times(num_samples: int, fps: float) -> np.ndarray:
    """Return sample timestamps in seconds for uniformly sampled motion data."""
    if fps <= 0.0:
        raise ValueError(f"FPS must be positive, got {fps}")
    return np.arange(num_samples, dtype=np.float64) / fps


def fourier_design_matrix(
    time_s: np.ndarray,
    order: int = FOURIER_ORDER,
    period_s: float | None = None,
) -> np.ndarray:
    if period_s is None:
        period_s = time_s.size / DEFAULT_FPS
    if period_s <= 0.0:
        raise ValueError(f"Fourier period must be positive, got {period_s}")

    columns = [np.ones(time_s.size)]
    for harmonic in range(1, order + 1):
        angle = 2.0 * np.pi * harmonic * time_s / period_s
        columns.append(np.cos(angle))
        columns.append(np.sin(angle))
    return np.column_stack(columns)


def evaluate_fourier(
    coefficients: np.ndarray,
    time_s: np.ndarray,
    period_s: float,
) -> np.ndarray:
    return fourier_design_matrix(time_s, (len(coefficients) - 1) // 2, period_s) @ coefficients


def fit_fourier(
    data: np.ndarray,
    time_s: np.ndarray,
    period_s: float,
    order: int = FOURIER_ORDER,
) -> tuple[np.ndarray, np.ndarray]:
    design = fourier_design_matrix(time_s, order, period_s)
    coefficients, *_ = np.linalg.lstsq(design, data, rcond=None)
    fitted = design @ coefficients
    return coefficients, fitted


def coefficient_dict(coefficients: np.ndarray) -> dict[str, float]:
    result = {"a0": float(coefficients[0])}
    for harmonic in range(1, (len(coefficients) - 1) // 2 + 1):
        result[f"a{harmonic}"] = float(coefficients[2 * harmonic - 1])
        result[f"b{harmonic}"] = float(coefficients[2 * harmonic])
    return result


def save_fitted_csv(
    output_path: Path,
    joint_names: list[str],
    time_s: np.ndarray,
    original: np.ndarray,
    fitted: np.ndarray,
) -> None:
    residual = original - fitted
    header = (
        ["time_s"]
        + joint_names
        + [f"{name}_fit" for name in joint_names]
        + [f"{name}_residual" for name in joint_names]
    )
    table = np.column_stack([time_s, original, fitted, residual])

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(table)


def save_plot(
    output_path: Path,
    joint_names: list[str],
    time_s: np.ndarray,
    original: np.ndarray,
    fitted: np.ndarray,
    order: int,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axes = plt.subplots(len(joint_names), 1, figsize=(10, 8), sharex=True)
    axes = np.atleast_1d(axes)

    for axis, name, y, y_fit in zip(axes, joint_names, original.T, fitted.T):
        axis.plot(time_s, y, label="raw", linewidth=1.2)
        axis.plot(time_s, y_fit, label=f"fourier_order_{order}", linewidth=1.2)
        axis.set_ylabel(name)
        axis.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--order", type=int, default=FOURIER_ORDER)
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help="Sampling rate of the input trajectories in frames per second.",
    )
    parser.add_argument(
        "--names",
        nargs=4,
        default=DEFAULT_JOINT_NAMES,
        help="Names for the four fitted columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_motion_table(args.input)
    if data.shape[1] != 4:
        raise ValueError(f"Expected exactly 4 columns, got {data.shape[1]} columns.")

    time_s = sample_times(data.shape[0], args.fps)
    period_s = data.shape[0] / args.fps

    coefficients, fitted = fit_fourier(data, time_s, period_s, args.order)
    residual = data - fitted
    rmse = np.sqrt(np.mean(residual**2, axis=0))
    max_abs_error = np.max(np.abs(residual), axis=0)

    output_dir = args.output_dir or args.input.with_name("fourier_fit")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_fitted_csv(output_dir / "fitted_head_neck.csv", args.names, time_s, data, fitted)

    payload = {
        "input": str(args.input),
        "num_samples": int(data.shape[0]),
        "fps": float(args.fps),
        "duration_s": float(period_s),
        "order": int(args.order),
        "model": "y(t)=a0+sum_k(a_k*cos(2*pi*k*t/T)+b_k*sin(2*pi*k*t/T))",
        "columns": {},
    }
    for idx, name in enumerate(args.names):
        payload["columns"][name] = {
            "coefficients": coefficient_dict(coefficients[:, idx]),
            "rmse": float(rmse[idx]),
            "max_abs_error": float(max_abs_error[idx]),
        }

    with (output_dir / "fourier_coefficients.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)

    plot_saved = save_plot(output_dir / "fourier_fit.png", args.names, time_s, data, fitted, args.order)

    print(f"Loaded {data.shape[0]} samples x {data.shape[1]} columns from {args.input}")
    print(f"Using fps={args.fps:g}, duration={period_s:.6f}s")
    print(f"Saved fitted CSV: {output_dir / 'fitted_head_neck.csv'}")
    print(f"Saved coefficients: {output_dir / 'fourier_coefficients.json'}")
    if plot_saved:
        print(f"Saved plot: {output_dir / 'fourier_fit.png'}")
    else:
        print("matplotlib is not installed; skipped plot output.")
    for name, error in zip(args.names, rmse):
        print(f"{name}: RMSE={error:.6f}")


if __name__ == "__main__":
    main()
