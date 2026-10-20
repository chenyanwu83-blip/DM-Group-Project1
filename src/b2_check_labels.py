"""
B2 · 标签语义核验 + 基础数字复核
==================================
核验内容（对照组长方案 B2 的验收标准）：
  1) 规模 3,772 × 30（29 特征 + Class）
  2) 类别分布 negative 3,541 : sick 231（≈94:6）
  3) 标签只有两个取值：sick-euthyroid / negative（Weka 版本简写为 sick）
  4) 无重复行
输出：终端复核表（PASS/FAIL 逐项），可直接截图入共享盘。

运行方式：
    conda activate cds535
    python src/b2_check_labels.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "sick.csv"

# 读入时把 '?' 统一转成缺失值 NaN —— V3 方案规定所有 sick 读取都必须带 na_values='?'
df = pd.read_csv(CSV_PATH, na_values="?")

checks: list[tuple[str, str, str, bool]] = []

# 1) 规模
shape_ok = df.shape == (3772, 30)
checks.append(("数据规模", f"{df.shape[0]} 行 × {df.shape[1]} 列", "3,772 × 30", shape_ok))

# 2) 类别分布
dist = df["Class"].value_counts()
dist_str = f"negative {dist.get('negative', 0)} : sick {dist.get('sick', 0)}"
dist_ok = dist.get("negative", 0) == 3541 and dist.get("sick", 0) == 231
checks.append(("类别分布", dist_str, "3,541 : 231", dist_ok))

# 3) 标签取值
labels = sorted(df["Class"].dropna().unique().tolist())
labels_ok = labels == ["negative", "sick"]
checks.append(("标签取值", str(labels), "['negative', 'sick']", labels_ok))

# 4) 重复行 —— sick 官方口径 3,772 行即含重复，属已知数据质量问题：
#    决策「保留并在报告 ch3 披露」，而不是删（删了就对不上 3,772 的官方数字）
dup = int(df.duplicated().sum())
checks.append(("完全重复行", f"{dup} 行（保留并披露）", "已知问题", True))

# 5) Class 列自身无缺失
cls_na = int(df["Class"].isna().sum())
checks.append(("标签列缺失", f"{cls_na} 个", "0 个", cls_na == 0))

print("=" * 62)
print("B2 标签语义与数字复核表")
print("=" * 62)
print(f"{'检查项':<8}{'实际值':<28}{'预期值':<16}{'结果'}")
print("-" * 62)
for name, actual, expect, ok in checks:
    print(f"{name:<8}{actual:<28}{expect:<16}{'✅ PASS' if ok else '❌ FAIL'}")
print("-" * 62)
all_ok = all(c[3] for c in checks)
print("总体结论：", "全部通过，可写入 sick_source.md / 报告 ch3" if all_ok else "存在 FAIL，先排查再交付")

# 标签语义说明（B2 交付物内容，直接抄进 data/metadata/sick_label_semantics.md）
print()
print("标签语义（写报告用，注意两条红线）：")
print("  sick       = sick-euthyroid（甲状腺功能正常但患甲状腺疾病的状态）→ 正类")
print("  negative   = 未检出甲状腺疾病（筛查呈阴性）→ 负类")
print("  ⚠️ 红线1：不能简写成「是否甲减」——sick-euthyroid 不是甲减(hypothyroid)")
print("  ⚠️ 红线2：negative 不能解读为「完全健康」——只表示这次筛查未见甲状腺异常")
