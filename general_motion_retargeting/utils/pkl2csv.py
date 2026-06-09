import pickle
import numpy as np
import pandas as pd

input_pkl = "raw_smpl/gvhmrmotion.pkl"
output_csv = "motion.csv"

with open(input_pkl, "rb") as f:
    data = pickle.load(f)

root_pos = data["root_pos"]     # (T,3)
root_rot = data["root_rot"]     # (T,4)
dof_pos  = data["dof_pos"]      # (T,29)

# 简单校验
T = root_pos.shape[0]
assert root_rot.shape[0] == T
assert dof_pos.shape[0] == T

# 拼接成一行 36 列
motion = np.concatenate([root_pos, root_rot, dof_pos], axis=1)

# 保存为 CSV
df = pd.DataFrame(motion)
df.to_csv(output_csv, index=False, header=False)

print("Saved:", output_csv)
print("Total frames:", T)
print("Columns:", df.shape[1])
