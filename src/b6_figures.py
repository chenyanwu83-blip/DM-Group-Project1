"""
B6 · sick 三张探索图（英文图面 → 直接进英文报告；同时存 PDF + PNG）
====================================================================
  ① fig_b_class.pdf    类别分布条形图（negative 3,541 : sick 231 ≈ 94:6）
  ② fig_b_missing.pdf  逐列缺失率条形图（TBG=100% 用红色标出）
  ③ fig_b_box.pdf      TSH、FTI 按类别箱线图（TSH 极右偏 → 对数轴）

图注（B6 交付物，直接译入报告；每张图 ≥2 句解读，满足课程"图必须解读"要求）
已写在本文件末尾的 FIGURE_CAPTIONS 里，运行时一并打印。

运行方式：
    conda activate cds535
    python src/b6_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # 无窗口环境也能出图（VSCode 里更稳）
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
CSV_PATH = ROOT / "data" / "raw" / "sick.csv"

df = pd.read_csv(CSV_PATH, na_values="?")
FIG_DIR.mkdir(exist_ok=True)

# ── 图① 类别分布 ──────────────────────────────────────────────
counts = df["Class"].value_counts()
fig, ax = plt.subplots(figsize=(5, 3.6))
bars = ax.bar(counts.index, counts.values,
              color=["#4C72B0", "#C44E52"], width=0.55)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 40, f"{v:,}",
            ha="center", fontsize=11)
ax.set_ylabel("Number of instances")
ax.set_title("Class distribution of sick (3,772 instances)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_b_class.pdf")
fig.savefig(FIG_DIR / "fig_b_class.png", dpi=150)
plt.close(fig)

# ── 图② 逐列缺失率（只画有缺失的列，TBG 标红）────────────────
na_pct = (df.isna().mean() * 100).sort_values()
na_pct = na_pct[na_pct > 0]
colors = ["#C44E52" if c == "TBG" else "#4C72B0" for c in na_pct.index]
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(na_pct.index, na_pct.values, color=colors)
for i, v in enumerate(na_pct.values):
    ax.text(v + 1.2, i, f"{v:.1f}%", va="center", fontsize=9)
ax.set_xlabel("Missing rate (%)")
ax.set_xlim(0, 112)
ax.set_title("Per-column missing rate of sick (TBG is 100% missing)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_b_missing.pdf")
fig.savefig(FIG_DIR / "fig_b_missing.png", dpi=150)
plt.close(fig)

# ── 图③ TSH / FTI 按类别箱线图（TSH 用对数轴压长尾）───────────
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
groups = [df.loc[df["Class"] == c, "TSH"].dropna() for c in ("negative", "sick")]
axes[0].boxplot(groups, tick_labels=["negative", "sick"])
axes[0].set_yscale("log")
axes[0].set_ylabel("TSH (log scale)")
axes[0].set_title("TSH by class")
groups = [df.loc[df["Class"] == c, "FTI"].dropna() for c in ("negative", "sick")]
axes[1].boxplot(groups, tick_labels=["negative", "sick"])
axes[1].set_ylabel("FTI")
axes[1].set_title("FTI by class")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.suptitle("Distributions of TSH and FTI by class", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_b_box.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "fig_b_box.png", dpi=150, bbox_inches="tight")
plt.close(fig)

FIGURE_CAPTIONS = """
图注（译成英文写入 Overleaf 的 sec3b_sick.tex，配合 \\caption{}）：

Fig 1  Class distribution of sick: 3,541 negative vs 231 sick-euthyroid
       (≈94:6). The severe imbalance explains why accuracy alone is
       misleading — always predicting "negative" already yields 93.9%.

Fig 2  Per-column missing rate. TBG is 100% missing (the test was never
       performed; its paired flag TBG_measured is constantly 'f'), so the
       column is removed rather than imputed. T3 has the second-highest
       rate (20.4%); after removing TBG, 1,129 rows (29.9%) still contain
       at least one missing value, which motivates fold-internal
       imputation instead of row deletion (see comparison experiment B4).

Fig 3  TSH (log scale) and FTI by class. The two classes overlap heavily
       (median TSH 1.4 vs 1.5; median FTI 107 vs 108): neither lab value
       alone separates the classes, so the discriminative signal must
       come from combinations of attributes — a motivation for
       multivariate models rather than single-threshold screening.
"""

print("✅ 三张图已保存到 figures/：")
for f in sorted(FIG_DIR.glob("fig_b_*")):
    print("   ", f.name)
print(FIGURE_CAPTIONS)
