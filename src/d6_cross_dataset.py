"""CDS535 D6: cross-dataset comparison table + winner-flip analysis inputs.

Input  : C5 t-test outputs (results/ttest/key_comparisons_v1.csv, raw+Holm p),
         C11 pre-specified ensemble tests (ensemble_2pairs_prespecified_v1.csv),
         D4 tables (all_metrics_v1.csv / ensemble_metrics_v1.csv, champions),
         raw data + split_manifest + B missing audit (dataset profile).
Output : results/cross_dataset_comparison_v1.csv  (strategy effect per dataset,
         p-values with significance marks + flip classification)
         results/d6_dataset_profile_v1.csv        (four-dimension profile)

Run from repository root:
    python src/d6_cross_dataset.py

Presentation decisions delegated to D6 by C5's record (星号=raw p @ alpha=0.05,
Holm reported alongside, NB(2003)-corrected t throughout) and D1 ("注明多重比
较") are implemented here and documented in docs/D6_跨数据集对比与赢家翻转_v1.0.md.
No model is trained and no p-value is recomputed: C5/C11 outputs are the single
source; only the flip classification and profile stats are derived here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODELS = ("dt", "rf", "lr", "svc")
STRATEGIES = ("smote", "cost", "threshold")
METRICS = ("f1", "recall", "accuracy")
ALPHA = 0.05


def flip_class(d_cg: float, p_cg: float, d_sk: float, p_sk: float) -> str:
    """Pre-registered classification of one strategy effect across datasets."""
    sig_cg, sig_sk = p_cg < ALPHA, p_sk < ALPHA
    if d_cg == 0.0 and d_sk == 0.0:
        return "no_effect_both"
    if d_sk == 0.0:
        return "creditg_only_effect" if d_cg > 0 else "creditg_only_loss"
    if d_cg == 0.0:
        return "sick_only_effect" if d_sk > 0 else "sick_only_loss"
    if np.sign(d_cg) != np.sign(d_sk):
        return "sign_flip_sig" if (sig_cg or sig_sk) else "sign_flip_ns"
    direction = "gain" if d_cg > 0 else "loss"
    return f"same_direction_{direction}_sig" if (sig_cg or sig_sk) \
        else f"same_direction_{direction}_ns"


def build_comparison():
    key = pd.read_csv(RESULTS / "ttest" / "key_comparisons_v1.csv")
    rows = []
    for metric in METRICS:
        for model in MODELS:
            for strategy in STRATEGIES:
                name = f"{strategy}_vs_plain"
                cg = key[(key.dataset == "credit-g") & (key.metric == metric)
                         & (key.comparison == f"creditg_{model}_{name}")].iloc[0]
                sk = key[(key.dataset == "sick") & (key.metric == metric)
                         & (key.comparison == f"sick_{model}_{name}")].iloc[0]
                rows.append({
                    "model": model, "strategy": strategy, "metric": metric,
                    "delta_creditg": round(cg.mean_diff, 4),
                    "p_creditg": round(cg.p_raw, 4),
                    "sig_creditg": cg.sig if isinstance(cg.sig, str) else "",
                    "delta_sick": round(sk.mean_diff, 4),
                    "p_sick": round(sk.p_raw, 4),
                    "sig_sick": sk.sig if isinstance(sk.sig, str) else "",
                    "flip_class": flip_class(cg.mean_diff, cg.p_raw,
                                              sk.mean_diff, sk.p_raw),
                })
    return pd.DataFrame(rows)


def build_profile():
    manifest = json.loads(
        (RESULTS / "split_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for dataset, csv in (("credit-g", "credit-g.csv"), ("sick", "sick.csv")):
        df = pd.read_csv(ROOT / "data" / "raw" / csv, na_values="?")
        target = df.columns[-1]
        pos = {"credit-g": "bad", "sick": "sick"}[dataset]
        features = df.drop(columns=[target])
        all_missing = [c for c in features if features[c].isna().all()]
        miss_rows = (features.drop(columns=all_missing).isna().any(axis=1).mean())
        counts = df[target].value_counts()
        rows.append({
            "dataset": dataset,
            "n_rows": manifest[dataset]["n_rows"],
            "n_features": len(features.columns),
            "n_numeric": int(sum(t.kind in "if" for t in features.dtypes)),
            "n_categorical": int(sum(t.kind == "O" for t in features.dtypes)),
            "n_positive": int(counts[pos]),
            "pos_rate": round(counts[pos] / len(df), 4),
            "imbalance_neg_pos": round(counts[~counts.index.isin([pos])].iloc[0]
                                       / counts[pos], 1),
            "all_missing_cols": ";".join(all_missing) or "none",
            "missing_row_rate_excl_all_missing": round(float(miss_rows), 4),
            "top_missing_cols": ";".join(
                f"{c}={v:.1%}" for c, v in features.isna().mean()
                .sort_values(ascending=False).head(3).items() if v > 0) or "none",
        })
    return pd.DataFrame(rows)


def main():
    comp = build_comparison()
    prof = build_profile()

    out_comp = RESULTS / "cross_dataset_comparison_v1.csv"
    out_prof = RESULTS / "d6_dataset_profile_v1.csv"
    comp.to_csv(out_comp, index=False)
    prof.to_csv(out_prof, index=False)
    print(f"saved: {out_comp.relative_to(ROOT)} ({len(comp)} rows)")
    print(f"saved: {out_prof.relative_to(ROOT)} ({len(prof)} rows)")

    print("\nprofile:")
    print(prof.to_string(index=False))

    for metric in METRICS:
        sub = comp[comp.metric == metric]
        print(f"\n=== strategy effect flip table · {metric} ===")
        print(sub[["model", "strategy", "delta_creditg", "sig_creditg",
                   "delta_sick", "sig_sick", "flip_class"]].to_string(index=False))

    # champions + ensemble 2-pair tests, printed for the D6 record
    allm = pd.read_csv(RESULTS / "all_metrics_v1.csv")
    ens = pd.read_csv(RESULTS / "ensemble_metrics_v1.csv")
    cv = allm[allm.protocol == "cv10"]
    print("\n=== champions (cv10 f1) ===")
    for dataset in ("credit-g", "sick"):
        best = cv[cv.dataset == dataset].nlargest(1, "f1_mean").iloc[0]
        print(f"{dataset}: main-matrix champion = {best.config_id} "
              f"f1={best.f1_mean:.4f}±{best.f1_std:.4f}")
        ebest = ens[(ens.dataset == dataset) & (ens.protocol == "cv10")].nlargest(
            1, "f1_mean").iloc[0]
        print(f"{dataset}: ensemble best        = {ebest.config_id} "
              f"f1={ebest.f1_mean:.4f}±{ebest.f1_std:.4f} "
              f"(runtime {ebest.runtime_sec:.1f}s)")
    print("\n=== C11 pre-specified ensemble tests (verbatim p-values) ===")
    two = pd.read_csv(RESULTS / "ttest" / "ensemble_2pairs_prespecified_v1.csv")
    print(two[["dataset", "comparison", "mean_a", "mean_b", "mean_diff",
               "p_raw", "sig"]].to_string(index=False))


if __name__ == "__main__":
    main()
