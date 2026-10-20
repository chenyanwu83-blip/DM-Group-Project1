"""
B3 · sick 缺失审计（本数据集的核心工作）
=========================================
产出（组长方案 B3 交付物）：
  1) results/stats/sick_missing_audit.csv —— 逐列缺失数/缺失率排序表
  2) 终端打印审计摘要：总缺失、TBG 机制、measured 配对核查
  3) 处理决策（删全缺列 + 折内插补）——直接抄进审计报告

运行方式：
    conda activate cds535
    python src/b3_missing_audit.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "sick.csv"
STATS_DIR = ROOT / "results" / "stats"

df = pd.read_csv(CSV_PATH, na_values="?")
df.columns = [c.replace(" measured", "_measured") for c in df.columns]  # 兼容组仓库空格版列名

# ── 1) 逐列缺失率表 ─────────────────────────────────────────
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
# TBG 100% 缺失会让「每一行都带缺失」——统计受影响行数前先剔除全缺列才有意义
rows_with_missing = int(df.drop(columns=all_missing_cols).isna().any(axis=1).sum())

print("=" * 66)
print("B3 sick 缺失审计报告（原始文件 data/raw/sick.csv）")
print("=" * 66)
print(f"总单元格数        : {total_cells:,}")
print(f"总缺失单元格      : {total_missing:,}（占全部单元格 {total_missing / total_cells * 100:.2f}%）")
print(f"100% 缺失的列     : {all_missing_cols}")
print(f"剔除全缺列后仍有缺失的行 : {rows_with_missing:,} / {len(df):,}"
      f"（{rows_with_missing / len(df) * 100:.1f}% 的行受缺失影响）")
print()
print("缺失率前 12 列：")
print(audit.head(12).to_string(index=False))

# ── 2) measured 标记列与检验值列的配对核查 ──────────────────
# 机制：每项化验有配对两列——X_measured(t/f) 与化验值 X。
#       X_measured=f 表示该次没做这项化验 → X 必然缺失。
print()
print("=" * 66)
print("measured 标记列 × 检验值列 配对核查（测量机制验证）")
print("=" * 66)
labs = ["TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
pair_rows = []
for lab in labs:
    m, v = df[f"{lab}_measured"], df[lab]
    r1 = int(((m == "f") & v.isna()).sum())     # 没做化验 → 值缺失（合理）
    r2 = int(((m == "f") & v.notna()).sum())    # 没做化验 却有值（异常，应为0）
    r3 = int(((m == "t") & v.isna()).sum())     # 做了化验 却没记录（数据质量问题）
    r4 = int(((m == "t") & v.notna()).sum())    # 做了化验 有值（合理）
    pair_rows.append([f"{lab}_measured", r1, r2, r3, r4])
pair_tb = pd.DataFrame(pair_rows, columns=[
    "配对列", "没做→缺(正常)", "没做→有值(异常)", "做了→缺(异常)", "做了→有值(正常)"])
print(pair_tb.to_string(index=False))

tbg_m_all_f = (df["TBG_measured"] == "f").all()
print(f"\nTBG_measured 是否全部为 f：{'是' if tbg_m_all_f else '否'}"
      f"（TBG 从未做过化验 → TBG 列 100% 缺失，机制自洽）")

# ── 3) 处理决策（写入审计报告 + B5 规格单）──────────────────
print()
print("=" * 66)
print("处理决策（B3 结论，交 C 嵌入 imblearn Pipeline）")
print("=" * 66)
print(f"① 删除 100% 全缺列：{all_missing_cols}")
print("   理由：全列无任何观测值，任何插补都等于编造数据，不可插补。")
print("   实现：df.drop(columns=['TBG'])（TBG_measured 恒为 f，同样删除）")
print("② 其余列折内插补：数值列均值 / 名义列众数（SimpleImputer）")
print("   理由：缺失源于「该项化验没做」（MAR），插补优于删行——删行会损失")
print(f"   {rows_with_missing / len(df) * 100:.1f}% 的样本，且病人数据每行都珍贵（B4 有实验证明）。")
print("   防泄漏：插补器放进 Pipeline，只在每折训练部分 fit（详见 B5 规格单）。")
print()
print(f"✅ 已保存逐列缺失表：{STATS_DIR / 'sick_missing_audit.csv'}")
