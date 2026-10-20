# -*- coding: utf-8 -*-
"""
C2 交付物: 两张统一流水线结构图 → figures/pipeline_creditg.png / pipeline_sick.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 输出锚定到脚本所在目录(figures/), 与运行时工作目录无关 —— PyCharm/终端/任意 cwd 均可
BASE = Path(__file__).resolve().parent

# macOS 中文字体(DejaVu Sans 无 CJK 字形)
matplotlib.rcParams["font.sans-serif"] = [
    "PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "Heiti SC"]
matplotlib.rcParams["axes.unicode_minus"] = False


def draw(path, title, pre_label, label_box):
    fig, ax = plt.subplots(figsize=(9.5, 2.8))
    boxes = [label_box, pre_label,
             "SMOTE（可选）\n仅训练折执行", "分类器\nDT / RF / LR / SVC"]
    n = len(boxes)
    xs = np.linspace(0.5, n - 0.5, n)
    for x, b in zip(xs, boxes):
        ax.add_patch(mp.FancyBboxPatch((x - 0.46, 0.38), 0.92, 0.5,
                     boxstyle="round,pad=0.03", fc="#eef3f8", ec="#5f8d87"))
        ax.text(x, 0.63, b, ha="center", va="center", fontsize=9)
    for i in range(n - 1):
        ax.annotate("", xy=(xs[i + 1] - 0.48, 0.63), xytext=(xs[i] + 0.48, 0.63),
                    arrowprops=dict(arrowstyle="->", color="#b3766e", lw=1.6))
    ax.text(n / 2, 0.10, "验证折：只做 transform，不拟合、不重采样（D1 防泄漏红线）",
            ha="center", fontsize=8.5, color="#a5503f")
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print("saved:", path)


if __name__ == "__main__":
    draw(BASE / "pipeline_creditg.png", "credit-g 统一流水线（imblearn.Pipeline）",
         "ColumnTransformer 预处理\n线性: OneHot+StandardScaler\n树: Ordinal+数值 passthrough",
         label_box="原始数据 X, y\n(字符串标签 bad/good)")
    draw(BASE / "pipeline_sick.png", "sick 统一流水线（imblearn.Pipeline）",
         "ColumnTransformer 预处理\n折内插补(数值mean/名义众数)\n线性: OneHot+Scaler / 树: Ordinal",
         label_box="原始数据 X, y\n(删 TBG 两列后)\n(字符串标签 sick/negative)")
