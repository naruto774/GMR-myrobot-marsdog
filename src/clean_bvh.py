import re
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R


# load_dog_bvh_file() later applies D1(Y-up) -> MuJoCo(Z-up):
#   [x, y, z] -> [x, -z, y]
# Blender-exported d1-tpose.bvh is already in that transformed basis, so the
# cleaner maps it back to the original D1 BVH basis before LAFAN FK reads it.
BLENDER_TO_D1_MATRIX = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]
)


def _rotation_channels(channel_tokens):
    """Return the rotation channel order from a BVH CHANNELS line."""
    rot_channels = [token for token in channel_tokens if token.endswith("rotation")]
    if len(rot_channels) != 3:
        raise ValueError(f"Expected 3 rotation channels, got: {channel_tokens}")
    return rot_channels


def _format_offset(line, matrix):
    offset_match = re.match(
        r"(\s*OFFSET\s+)([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)(.*)", line
    )
    if not offset_match:
        return line

    prefix, x, y, z, suffix = offset_match.groups()
    corrected = matrix @ np.array([float(x), float(y), float(z)])
    return (
        f"{prefix}{corrected[0]:.6f} {corrected[1]:.6f} {corrected[2]:.6f}"
        f"{suffix}\n"
    )


def _channel_rotation_order(rot_channels):
    return "".join(channel[0].upper() for channel in rot_channels)


def _convert_euler_basis(euler_deg, rot_channels, matrix):
    """Change coordinates for BVH Euler channels while preserving order."""
    order = _channel_rotation_order(rot_channels)
    rotation = R.from_euler(order, euler_deg, degrees=True)
    converted = R.from_matrix(matrix @ rotation.as_matrix() @ matrix.T)
    return converted.as_euler(order, degrees=True)


def _convert_position(pos, matrix):
    return matrix @ np.asarray(pos, dtype=float)


def clean_blender_bvh(input_path, output_path, axis_convert=True):
    """Convert Blender mixed-channel BVH to Root(6)+joint(3) LAFAN layout."""
    with open(input_path, "r") as f:
        lines = f.readlines()

    output_lines = []
    motion_mode = False
    channel_specs = []  # 记录每个关节原始通道和旋转顺序，用于切 MOTION 数据
    axis_matrix = BLENDER_TO_D1_MATRIX if axis_convert else np.eye(3)

    # 1. 净化 HIERARCHY 结构
    for line_no, line in enumerate(lines, start=1):
        if "MOTION" in line:
            motion_mode = True
            output_lines.append(line)
            continue

        if not motion_mode:
            channel_match = re.match(r"(\s*)CHANNELS\s+(\d+)\s+(.+)", line)
            if channel_match:
                indent, channel_count_str, channel_tail = channel_match.groups()
                channel_count = int(channel_count_str)
                channel_tokens = channel_tail.split()
                rot_channels = _rotation_channels(channel_tokens)
                channel_specs.append((channel_count, rot_channels))

                # 第一个 CHANNELS 属于 ROOT，必须保留 6 通道。
                if len(channel_specs) == 1:
                    if channel_count != 6:
                        raise ValueError(
                            f"Root CHANNELS must be 6, got {channel_count} at line {line_no}"
                        )
                    output_lines.append(line)
                    continue

                # 子关节只保留旋转通道，丢弃 Blender bake 出来的局部平移通道。
                output_lines.append(f"{indent}CHANNELS 3 {' '.join(rot_channels)}\n")
            elif axis_convert and re.match(r"\s*OFFSET\s+", line):
                output_lines.append(_format_offset(line, axis_matrix))
            else:
                output_lines.append(line)
        else:
            # 2. 净化 MOTION 帧数据
            if "Frames:" in line or "Frame Time:" in line:
                output_lines.append(line)
            else:
                # 这一行是真正的浮点数帧数据
                frame_data = np.fromstring(line, sep=" ")
                if len(frame_data) == 0:
                    continue

                expected_input_len = sum(ch_num for ch_num, _ in channel_specs)
                if len(frame_data) != expected_input_len:
                    raise ValueError(
                        f"Frame at line {line_no} has {len(frame_data)} values, "
                        f"expected {expected_input_len} from HIERARCHY channels"
                    )

                cleaned_frame = []
                idx = 0
                # 根据 HIERARCHY 的记录，切除子关节多余的 pos 数据
                for i, (ch_num, rot_channels) in enumerate(channel_specs):
                    if i == 0:  # ROOT 关节，全保留
                        root_pos = _convert_position(frame_data[idx:idx+3], axis_matrix)
                        root_rot = _convert_euler_basis(
                            frame_data[idx+3:idx+6], rot_channels, axis_matrix
                        )
                        cleaned_frame.extend(root_pos)
                        cleaned_frame.extend(root_rot)
                        idx += 6
                    else:
                        if ch_num == 6:
                            # Blender 导出的 6 通道是 [pos_x, pos_y, pos_z, rot_z, rot_x, rot_y]
                            # 我们扔掉前 3 个 pos，只保留后 3 个 rot
                            joint_rot = _convert_euler_basis(
                                frame_data[idx+3:idx+6], rot_channels, axis_matrix
                            )
                            cleaned_frame.extend(joint_rot)
                            idx += 6
                        else:
                            joint_rot = _convert_euler_basis(
                                frame_data[idx:idx+3], rot_channels, axis_matrix
                            )
                            cleaned_frame.extend(joint_rot)
                            idx += 3

                expected_output_len = 6 + 3 * (len(channel_specs) - 1)
                if len(cleaned_frame) != expected_output_len:
                    raise ValueError(
                        f"Cleaned frame at line {line_no} has {len(cleaned_frame)} values, "
                        f"expected {expected_output_len}"
                    )

                # 将净化后的 66 通道数据转回字符串写入
                frame_str = " ".join([f"{num:.6f}" for num in cleaned_frame]) + "\n"
                output_lines.append(frame_str)

    with open(output_path, "w") as f:
        f.writelines(output_lines)

    print(
        "成功将混合通道的 BVH 净化为 "
        f"{6 + 3 * (len(channel_specs) - 1)} 通道格式！保存在: {output_path}"
    )
    if axis_convert:
        print("已执行坐标轴转换: Blender/Z-up -> D1 BVH/Y-up，供 load_dog_bvh_file() 再转 MuJoCo。")


def main():
    parser = argparse.ArgumentParser(
        description="Clean Blender-exported mixed-channel BVH for LAFAN FK."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("assets/bvh/d2-tpose.bvh"),
        help="Input Blender-exported BVH.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/bvh/d2-tpose-clean.bvh"),
        help="Output Root(6)+joint(3) BVH.",
    )
    parser.add_argument(
        "--no-axis-convert",
        action="store_true",
        help="Only clean channels; do not convert Blender/Z-up coordinates back to D1/Y-up.",
    )
    args = parser.parse_args()

    clean_blender_bvh(args.input, args.output, axis_convert=not args.no_axis_convert)


if __name__ == "__main__":
    main()
