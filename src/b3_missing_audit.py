"""
统计 sick 数据集逐列缺失率，保存到 results/stats/sick_missing_audit.csv，
并核查 measured 标记列与化验值列的配对关系。
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "sick.csv"
STATS_DIR = ROOT / "results" / "stats"

df = pd.read_csv(CSV_PATH, na_values="?")
df.columns = [c.replace(" measured", "_measured") for c in df.columns]

# 逐列缺失率
audit = (
    pd.DataFrame({
        "column": df.columns,
        "n_missing": df.isna().sum().values,
        "pct_missing": (df.isna().mean() * 100).round(2).values,
    })
    .sort_values("pct_missing", ascending=False)
    .reset_index(drop=True)
)
STATS_DIR.mkdir(parents=True, exist_ok=True)
audit.to_csv(STATS_DIR / "sick_missing_audit.csv", index=False)

total_cells = df.shape[0] * df.shape[1]
total_missing = int(df.isna().sum().sum())
all_missing_cols = audit.loc[audit["pct_missing"] == 100, "column"].tolist()
# 剔除全缺列后再统计受影响的行数
rows_with_missing = int(df.drop(columns=all_missing_cols).isna().any(axis=1).sum())

print("缺失审计")
print(f"总单元格数        : {total_cells:,}")
print(f"总缺失单元格      : {total_missing:,}（占全部单元格 {total_missing / total_cells * 100:.2f}%）")
print(f"100% 缺失的列     : {all_missing_cols}")
print(f"剔除全缺列后仍有缺失的行 : {rows_with_missing:,} / {len(df):,}"
      f"（{rows_with_missing / len(df) * 100:.1f}% 的行受缺失影响）")
print()
print("缺失率前 12 列：")
print(audit.head(12).to_string(index=False))

# measured 标记列与化验值列的配对核查
# 机制：X_measured=f 表示没做该项化验，X 应缺失
print()
print("配对核查")
labs = ["TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
pair_rows = []
for lab in labs:
    m, v = df[f"{lab}_measured"], df[lab]
    r1 = int(((m == "f") & v.isna()).sum())     # 没做且缺失
    r2 = int(((m == "f") & v.notna()).sum())    # 没做但有值
    r3 = int(((m == "t") & v.isna()).sum())     # 做了但缺失
    r4 = int(((m == "t") & v.notna()).sum())    # 做了且有值
    pair_rows.append([f"{lab}_measured", r1, r2, r3, r4])
pair_tb = pd.DataFrame(pair_rows, columns=[
    "配对列", "没做且缺(正常)", "没做但有值(异常)", "做了但缺(异常)", "做了且有值(正常)"])
print(pair_tb.to_string(index=False))

tbg_m_all_f = (df["TBG_measured"] == "f").all()
print(f"\nTBG_measured 是否全部为 f：{'是' if tbg_m_all_f else '否'}"
      f"（TBG 未做过该项化验，与 TBG 列 100% 缺失一致）")

# 处理决策
print()
print("处理决策")
print(f"① 删除 100% 全缺列：{all_missing_cols}")
print("   理由：全列无观测值，无法插补。")
print("   实现：df.drop(columns=['TBG'])（TBG_measured 恒为 f，同样删除）")
print("② 其余列折内插补：数值列均值 / 名义列众数（SimpleImputer）")
print("   理由：缺失源于该项化验未做；删行会损失")
print(f"   {rows_with_missing / len(df) * 100:.1f}% 的样本（B4 实验对比过两种方案）。")
print("   插补器放在 Pipeline 内，只在训练折上拟合。")
print()
print(f"已保存：{STATS_DIR / 'sick_missing_audit.csv'}")
