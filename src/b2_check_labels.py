"""
核验 sick 数据集的规模、类别分布和标签取值，打印复核表。
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "sick.csv"

# '?' 转为缺失值
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

# 4) 重复行（官方数据自带，保留）
dup = int(df.duplicated().sum())
checks.append(("重复行", f"{dup} 行", "官方口径含重复", True))

# 5) Class 列自身无缺失
cls_na = int(df["Class"].isna().sum())
checks.append(("标签列缺失", f"{cls_na} 个", "0 个", cls_na == 0))

print("标签与数字复核")
print(f"{'检查项':<8}{'实际值':<28}{'预期值':<16}{'结果'}")
print("-" * 62)
for name, actual, expect, ok in checks:
    print(f"{name:<8}{actual:<28}{expect:<16}{'通过' if ok else '未通过'}")
all_ok = all(c[3] for c in checks)
print("结论：", "全部通过" if all_ok else "存在未通过项")

# 标签语义说明
print()
print("标签语义：")
print("  sick       = sick-euthyroid（甲状腺功能正常但化验异常，多由其他疾病引起）为正类")
print("  negative   = 筛查未检出甲状腺异常 为负类")
print("  注1：sick-euthyroid 不是甲减(hypothyroid)")
print("  注2：negative 只表示筛查未检出，不等于完全健康")
