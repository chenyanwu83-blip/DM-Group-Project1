"""CDS535 D5: three core figure groups for the results chapter.

Fig 1  sick ROC + PR overlay, plain vs smote vs cost (RandomForest, cv10 pooled)
Fig 2  strategy gain vs class imbalance, dF1 / dRecall bars (model mean, cv10)
Fig 3  champion confusion matrices, counts + row-normalized dual view

Outputs (figures/, vector PDF canonical + PNG preview, plus data CSVs under
figures/data/ so every plotted number is checkable - E figure rule "record the
underlying result file"):
    fig_d_sick_roc_pr.pdf/.png          fig_d_strategy_gain.pdf/.png
    fig_d_confusion.pdf/.png            figures/data/*.csv, fig_d_sources.csv

Run from repository root:   python src/d5_figures.py
No model is trained; curves come from the archived cv10 predictions.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PREDS = RESULTS / "preds"
FIGS = ROOT / "figures"
FIG_DATA = FIGS / "data"

# palette: dataviz reference palette, light mode, fixed slot order
# (validated order; aqua <3:1 contrast -> relief rule: legend + data CSV)
C_PLAIN, C_SMOTE, C_COST = "#2a78d6", "#eb6834", "#1baf7a"   # slots 1-3
C_DF1, C_DREC = "#2a78d6", "#eb6834"                          # slots 1-2
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SURFACE = "#fcfcfb"
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#5598e7", "#3987e5",
             "#256abf", "#1c5cab", "#104281", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "PingFang SC", "Hiragino Sans GB"],
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved: figures/{name}.pdf/.png")
    plt.close(fig)


def fig1_sick_roc_pr():
    strategies = [("plain", C_PLAIN), ("smote", C_SMOTE), ("cost", C_COST)]
    rows, curves = [], {}
    for strat, _ in strategies:
        p = pd.read_csv(PREDS / f"sick_rf_{strat}_v1_run1_cv10_preds.csv")
        yb = (p["y_true"] == "sick").to_numpy()
        s = p["proba_pos"].to_numpy(dtype=float)
        fpr, tpr, _ = roc_curve(yb, s)
        rec, prec, _ = precision_recall_curve(yb, s)
        curves[strat] = (fpr, tpr, rec, prec)
        rows.append({"strategy": strat, "roc_auc": roc_auc_score(yb, s),
                     "pr_ap": average_precision_score(yb, s), "n": len(p)})
    summary = pd.DataFrame(rows)
    summary.to_csv(FIG_DATA / "fig_d_sick_roc_pr_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for ax, panel in zip(axes, ("roc", "pr")):
        for strat, color in strategies:
            fpr, tpr, rec, prec = curves[strat]
            if panel == "roc":
                auc = summary.loc[summary.strategy == strat, "roc_auc"].iloc[0]
                ax.plot(fpr, tpr, color=color, lw=2,
                        label=f"{strat} (AUC = {auc:.3f})")
            else:
                ap = summary.loc[summary.strategy == strat, "pr_ap"].iloc[0]
                ax.plot(rec, prec, color=color, lw=2,
                        label=f"{strat} (AP = {ap:.3f})")
        if panel == "roc":
            ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls=":", label="chance")
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("(a) ROC, sick · cv10 pooled", fontsize=10)
        else:
            prev = 231 / 3772
            ax.axhline(prev, color=INK2, lw=1, ls=":", label=f"prevalence {prev:.3f}")
            ax.set_xlabel("Recall (positive = sick)")
            ax.set_ylabel("Precision")
            ax.set_title("(b) Precision-Recall, sick · cv10 pooled", fontsize=10)
        ax.legend(frameon=False, fontsize=8.5)
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.03)
    fig.tight_layout()
    save(fig, "fig_d_sick_roc_pr")


def fig2_strategy_gain():
    comp = pd.read_csv(RESULTS / "cross_dataset_comparison_v1.csv")
    # credit-g and sick deltas live in the same comparison row; melt to long form
    rows = []
    for (strategy, metric), g in comp.groupby(["strategy", "metric"]):
        rows.append({"dataset": "credit-g", "strategy": strategy, "metric": metric,
                     "delta_mean": g["delta_creditg"].mean()})
        rows.append({"dataset": "sick", "strategy": strategy, "metric": metric,
                     "delta_mean": g["delta_sick"].mean()})
    agg = pd.DataFrame(rows)
    agg = agg[agg.metric.isin(("f1", "recall"))]
    agg.to_csv(FIG_DATA / "fig_d_strategy_gain.csv", index=False)

    strategies = ["smote", "cost", "threshold"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharey=True)
    for ax, (dataset, imb) in zip(axes, (("credit-g", "2.3:1"), ("sick", "15.3:1"))):
        x = np.arange(len(strategies))
        for i, (metric, color, label) in enumerate(
                (("f1", C_DF1, "ΔF1"), ("recall", C_DREC, "ΔRecall"))):
            vals = [agg[(agg.dataset == dataset) & (agg.strategy == s)
                        & (agg.metric == metric)].delta_mean.iloc[0]
                    for s in strategies]
            bars = ax.bar(x + (i - 0.5) * 0.38, vals, width=0.34, color=color,
                          label=label, zorder=3)
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:+.3f}", (b.get_x() + b.get_width() / 2, v),
                            xytext=(0, 3 if v >= 0 else -11),
                            textcoords="offset points", ha="center",
                            fontsize=7.5, color=INK2)
        ax.axhline(0, color=INK2, lw=1)
        ax.set_xticks(x, strategies)
        ax.set_title(f"{dataset} (neg:pos {imb})", fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Δ vs plain (mean of 4 models)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()
    save(fig, "fig_d_strategy_gain")


def fig3_confusion():
    allm = pd.read_csv(RESULTS / "all_metrics_v1.csv")
    champs = [("credit-g", "creditg_lr_cost_v1", "bad", ("good", "bad")),
              ("sick", "sick_rf_smote_v1", "sick", ("negative", "sick"))]
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.0))
    for r, (dataset, cid, pos, (neg, _)) in enumerate(champs):
        m = allm[(allm.config_id == cid) & (allm.protocol == "cv10")].iloc[0]
        cm = np.array([[m.TN, m.FP], [m.FN, m.TP]], dtype=float)  # rows=pred, cols=actual
        norm = cm / cm.sum(axis=1, keepdims=True)
        for c, (mat, fmt) in enumerate(((cm, "{:,.0f}"), (norm, "{:.1%}"))):
            ax = axes[r, c]
            ax.imshow(mat, cmap=CMAP, vmin=0, vmax=mat.max() if c == 0 else 1)
            for i in range(2):
                for j in range(2):
                    dark = mat[i, j] > 0.55 * mat.max()
                    ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                            fontsize=10, color="#ffffff" if dark else INK)
                    rows.append({"config_id": cid, "view": ("counts", "row_pct")[c],
                                 "pred": "neg" if i == 0 else "pos",
                                 "actual": "neg" if j == 0 else "pos",
                                 "value": mat[i, j]})
            ax.set_xticks([0, 1], [f"actual {neg}", f"actual {pos} (+)"], fontsize=9)
            ax.set_yticks([0, 1], [f"pred {neg}", f"pred {pos} (+)"], fontsize=9)
            ax.grid(visible=False)
            view = "counts" if c == 0 else "row-normalized"
            ax.set_title(f"{cid} · {view} · cv10 pooled", fontsize=9)
    pd.DataFrame(rows).to_csv(FIG_DATA / "fig_d_confusion.csv", index=False)
    fig.tight_layout()
    save(fig, "fig_d_confusion")


def main():
    FIG_DATA.mkdir(exist_ok=True)
    fig1_sick_roc_pr()
    fig2_strategy_gain()
    fig3_confusion()
    sources = pd.DataFrame([
        {"figure": "fig_d_sick_roc_pr",
         "source_files": "results/preds/sick_rf_{plain,smote,cost}_v1_run1_cv10_preds.csv",
         "params": "pooled cv10 predictions; model=RandomForest; positive=sick"},
        {"figure": "fig_d_strategy_gain",
         "source_files": "results/cross_dataset_comparison_v1.csv (D6, from C5 key_comparisons)",
         "params": "mean of 4 models; cv10; Δ = strategy − plain"},
        {"figure": "fig_d_confusion",
         "source_files": "results/all_metrics_v1.csv (D4; pooled TN/FP/FN/TP)",
         "params": "cv10 pooled; champions creditg_lr_cost_v1 / sick_rf_smote_v1"},
    ])
    sources.to_csv(FIGS / "fig_d_sources.csv", index=False)
    print("saved: figures/fig_d_sources.csv")


if __name__ == "__main__":
    main()
