"""A4｜credit-g 三张探索图 + 图注（角色 A 交付物）

figures/ 下产出（草图，E 后期统一编号样式）：
- fig1_class_distribution.png   类别分布
- fig2_checking_status_class.png checking_status × class 堆叠柱状
- fig3_numeric_hist.png          duration/credit_amount/age 直方图（bins=20）
- credit-g_fig_captions.md      图注（英文 caption 供 Overleaf + 中文解读，数字自动回填）
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "raw" / "credit-g.csv"
FIG_DIR = ROOT / "figures"

plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3})


def main() -> None:
    df = pd.read_csv(CSV)
    FIG_DIR.mkdir(exist_ok=True)

    # fig1: class distribution
    ax = df["class"].value_counts().plot(kind="bar", color=["#4c72b0", "#c44e52"], rot=0)
    ax.set_title("Class Distribution (credit-g)")
    ax.set_xlabel("class")
    ax.set_ylabel("count")
    for c in ax.containers:
        ax.bar_label(c)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_class_distribution.png")
    plt.close()

    # fig2: checking_status x class stacked
    ct = pd.crosstab(df["checking_status"], df["class"])
    bad_rate = (ct["bad"] / ct.sum(axis=1)).sort_values(ascending=False)
    ax = ct.loc[bad_rate.index].plot(kind="bar", stacked=True, color=["#4c72b0", "#c44e52"], rot=30)
    ax.set_title("checking_status vs class (stacked)")
    ax.set_xlabel("checking_status")
    ax.set_ylabel("count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_checking_status_class.png")
    plt.close()

    # fig3: numeric histograms
    cols = ["duration", "credit_amount", "age"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, col in zip(axes, cols):
        df[col].hist(bins=20, ax=ax, color="#55a868")
        ax.set_title(col)
    fig.suptitle("Numeric Attributes Distribution (bins=20)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_numeric_hist.png")
    plt.close()

    n = len(df)
    dist = df["class"].value_counts().to_dict()
    worst, best = bad_rate.index[0], bad_rate.index[-1]
    rate_lines = "\n".join(
        f"| {k} | {int(ct.loc[k].sum())} | {int(ct.loc[k]['bad'])} | {ct.loc[k]['bad'] / ct.loc[k].sum():.1%} |"
        for k in bad_rate.index
    )
    captions = f"""# credit-g 图注（A4 交付，数字由 src/a4_creditg_eda.py 自动生成）

## fig1_class_distribution.png
- **Caption (EN)**: Class distribution of the credit-g dataset: {dist['good']} good vs. {dist['bad']} bad ({dist['bad'] / n:.0%} positive rate), a 70:30 class imbalance.
- **解读**：正类 bad 占 {dist['bad'] / n:.0%}，属中度不平衡；多数类基线 accuracy=70%，是 RQ2「accuracy 何时失效」的下限参照。

## fig2_checking_status_class.png
- **Caption (EN)**: Stacked distribution of class by checking_status, sorted by bad rate. Overdrawn accounts ("{worst}") show the highest default rate ({bad_rate.iloc[0]:.1%}), while "{best}" the lowest ({bad_rate.iloc[-1]:.1%}).
- **解读**：支票账户状态与违约呈清晰梯度，与"账户余额越少、信用风险越高"的信贷直觉一致：

| checking_status | 样本数 | bad | bad 率 |
|---|---|---|---|
{rate_lines}

  bad 率从 {bad_rate.iloc[0]:.1%}（{worst}）单调降至 {bad_rate.iloc[-1]:.1%}（{best}），横跨整体 30% 基线两侧——验证 checking_status 为最强预测特征之一；报告 ch3 可引用该表。

## fig3_numeric_hist.png
- **Caption (EN)**: Histograms (bins=20) of duration, credit_amount and age. All three are right-skewed; credit_amount ranges from {df.credit_amount.min():.0f} to {df.credit_amount.max():.0f} DM.
- **解读**：三个数值属性均右偏（中位数 {df.credit_amount.median():.0f} DM ≪ 最大 {df.credit_amount.max():.0f} DM），支持 A5 规格单在等频离散化（quantile binning）中使用分位数切分而非等宽，避免高频箱塌缩。
"""
    (FIG_DIR / "credit-g_fig_captions.md").write_text(captions, encoding="utf-8")
    print("已生成 3 张图 + 图注：", FIG_DIR)
    print("\nbad 率排序：")
    print(bad_rate.map(lambda x: f"{x:.1%}").to_string())


if __name__ == "__main__":
    main()
