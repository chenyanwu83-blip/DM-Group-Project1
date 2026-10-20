"""CDS535 D3: plain baseline metrics table v1 + independent direction audit.

Input  : C3-2 plain baseline results (8 configs + 2 ZeroR refs, cv10 + holdout),
         D2 split assets (holdout_idx / cv_folds / split_manifest.json),
         D2 evaluate.py (metrics recomputed with the same locked functions).
Output : results/plain_metrics_v1.csv        (D3 deliverable 1)
         results/d3_direction_check_v1.csv   (D3 deliverable 2, per-check PASS/FAIL)

Run from repository root:
    python src/d3_plain_metrics.py

No model is trained here: every number is either read from C's CSVs or
recomputed from the archived predictions in results/preds/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import (  # noqa: E402
    DATASETS, METRICS, SUMMARY_COLUMNS, FOLD_COLUMNS, compute_metrics,
)

RESULTS = ROOT / "results"
PREDS = RESULTS / "preds"
CONFIGS = [
    # (config_id, dataset, model, strategy)
    ("creditg_dt_plain_v1", "credit-g", "DecisionTree", "plain"),
    ("creditg_rf_plain_v1", "credit-g", "RandomForest", "plain"),
    ("creditg_lr_plain_v1", "credit-g", "Logistic", "plain"),
    ("creditg_svc_plain_v1", "credit-g", "SVC", "plain"),
    ("sick_dt_plain_v1", "sick", "DecisionTree", "plain"),
    ("sick_rf_plain_v1", "sick", "RandomForest", "plain"),
    ("sick_lr_plain_v1", "sick", "Logistic", "plain"),
    ("sick_svc_plain_v1", "sick", "SVC", "plain"),
    ("creditg_zero_ref_v1", "credit-g", "ZeroR", "zero_ref"),
    ("sick_zero_ref_v1", "sick", "ZeroR", "zero_ref"),
]
PROTOCOLS = ("cv10", "holdout")
HOLDOUT_IDX = {"credit-g": "holdout_idx_creditg.csv", "sick": "holdout_idx_sick.csv"}
TOL = 1e-9


def check_row(checks, config_id, protocol, name, ok, detail=""):
    checks.append({"config_id": config_id, "protocol": protocol, "check": name,
                   "result": "PASS" if ok else "FAIL", "detail": detail})
    return bool(ok)


def close(a, b, tol=TOL):
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(float(a) - float(b)) <= tol


def audit_config(config_id, dataset, model, strategy, protocol, raw_labels,
                 manifest, checks):
    """Run all D3 checks for one config x protocol; return summary row."""
    spec = DATASETS[dataset]
    n_rows = manifest[dataset]["n_rows"]
    suffix = "" if protocol == "cv10" else "_holdout"
    summary = pd.read_csv(RESULTS / f"{config_id}_run1{suffix}_summary.csv")
    folds = pd.read_csv(RESULTS / f"{config_id}_run1{suffix}_folds.csv")
    preds = pd.read_csv(PREDS / f"{config_id}_run1_{protocol}_preds.csv")
    row = summary.iloc[0]
    ok_all = True

    # --- A. schema / meta ------------------------------------------------
    ok_all &= check_row(checks, config_id, protocol, "summary columns == D2 template",
                        list(summary.columns) == SUMMARY_COLUMNS)
    ok_all &= check_row(checks, config_id, protocol, "folds columns == D2 template",
                        list(folds.columns) == FOLD_COLUMNS)
    ok_all &= check_row(
        checks, config_id, protocol, "pos/neg label == D1 mapping",
        (row["pos_label"], row["neg_label"]) == (spec["positive"], spec["negative"]),
        f"pos={row['pos_label']}, neg={row['neg_label']}")
    ok_all &= check_row(checks, config_id, protocol, "seed == 42", row["seed"] == 42)
    ok_all &= check_row(
        checks, config_id, protocol, "data_sha256 == split_manifest",
        row["data_sha256"] == manifest[dataset]["sha256"])
    n_expected_folds = 10 if protocol == "cv10" else 1
    ok_all &= check_row(checks, config_id, protocol, "n_folds rows on disk",
                        len(folds) == n_expected_folds, f"{len(folds)}")
    ok_all &= check_row(checks, config_id, protocol, "protocol field",
                        row["protocol"] == protocol)

    # --- B. folds internal consistency ------------------------------------
    cm_sum = folds[["TN", "FP", "FN", "TP"]].sum()
    ok_all &= check_row(checks, config_id, protocol, "fold cm sums == n_eval",
                        bool((folds["TN"] + folds["FP"] + folds["FN"] + folds["TP"]
                              == folds["n_eval"]).all()))
    if dataset == "credit-g":
        cost_ok = np.allclose(
            folds["cost"], (5 * folds["FN"] + folds["FP"]) / folds["n_eval"])
        ok_all &= check_row(checks, config_id, protocol, "fold cost == (5FN+FP)/N",
                            bool(cost_ok))

    # summary aggregates recomputed from folds (D1: population SD, ddof=0)
    agg_ok, agg_bad = True, []
    for metric in METRICS:
        values = folds[metric].astype(float).to_numpy()
        finite = values[np.isfinite(values)]
        for stat, recomputed in (
                ("mean", finite.mean() if finite.size else np.nan),
                ("std", finite.std(ddof=0)
                 if protocol == "cv10" and finite.size == 10 else np.nan),
                ("n_valid", float(finite.size))):
            if not close(row[f"{metric}_{stat}"], recomputed):
                agg_ok = False
                agg_bad.append(f"{metric}_{stat}")
    ok_all &= check_row(checks, config_id, protocol,
                        "summary mean/std/n_valid == recomputed from folds", agg_ok,
                        ";".join(agg_bad))
    pooled_ok = all(int(cm_sum[k]) == int(row[k]) for k in ("TN", "FP", "FN", "TP"))
    ok_all &= check_row(checks, config_id, protocol, "pooled cm == sum of folds", pooled_ok,
                        f"TN={cm_sum['TN']},FP={cm_sum['FP']},"
                        f"FN={cm_sum['FN']},TP={cm_sum['TP']}")
    if protocol == "holdout":
        ok_all &= check_row(checks, config_id, protocol, "holdout std blank",
                            all(pd.isna(row[f"{m}_std"]) for m in METRICS
                                if not (m == "cost" and dataset == "sick")))

    # --- C. independent direction audit from preds ------------------------
    # C1. row coverage: cv10 predicts every row once; holdout == locked indices
    idx = preds["row_idx"].to_numpy()
    if protocol == "cv10":
        covered = np.array_equal(np.sort(idx), np.arange(n_rows))
        detail = f"{len(idx)} rows"
    else:
        locked = pd.read_csv(RESULTS / HOLDOUT_IDX[dataset])["row_idx"].to_numpy()
        covered = set(idx.tolist()) == set(locked.tolist()) and len(idx) == len(locked)
        detail = f"{len(idx)} rows vs locked {len(locked)}"
    ok_all &= check_row(checks, config_id, protocol, "preds cover locked rows exactly",
                        bool(covered), detail)

    # C2. y_true in preds == raw data labels at the same rows (label direction)
    truth_ok = bool((preds["y_true"].to_numpy()
                     == raw_labels[dataset][idx].to_numpy()).all())
    ok_all &= check_row(checks, config_id, protocol,
                        "preds y_true == raw data labels at row_idx", truth_ok)

    # C3. positive probability validity + boundary consistency at tau=0.5.
    # SVC(probability=True) Platt-scales proba independently of the decision
    # function, and RF majority vote vs mean proba can tie at exactly 0.5, so
    # "pred=neg while proba slightly >= 0.5" is a benign recorded exception
    # (band 0.10). pred=pos while proba < 0.5 is always a direction failure.
    proba = preds["proba_pos"].to_numpy(dtype=float)
    is_pos = (preds["y_pred"] == spec["positive"]).to_numpy()
    valid = bool(np.isfinite(proba).all() and ((proba < 0) | (proba > 1)).sum() == 0)
    hard_violation = int((is_pos & (proba < 0.5)).sum())
    neg_over = ~is_pos & (proba >= 0.5)
    benign = int((neg_over & (proba <= 0.6)).sum())
    outside = int((neg_over & (proba > 0.6)).sum())
    max_dev = float(np.max(proba[neg_over] - 0.5)) if neg_over.any() else 0.0
    proba_ok = valid and hard_violation == 0 and outside == 0
    ok_all &= check_row(
        checks, config_id, protocol,
        "proba_pos valid; y_pred==pos => proba>=0.5; neg-side excess within 0.10 band",
        proba_ok,
        f"pos-side violations={hard_violation}; benign band rows={benign} "
        f"(max proba-0.5={max_dev:.4f}); outside band={outside}")

    # C4. probability direction: scores separate the classes (skip ZeroR)
    if strategy != "zero_ref":
        tb = (preds["y_true"] == spec["positive"]).to_numpy()
        sep = proba[tb].mean() > proba[~tb].mean()
        ok_all &= check_row(checks, config_id, protocol,
                            "mean proba_pos(true pos) > mean proba_pos(true neg)",
                            bool(sep),
                            f"{proba[tb].mean():.4f} vs {proba[~tb].mean():.4f}")

    # C5. pooled metrics recomputed with D2 evaluate.py == summary pooled row
    recomputed = compute_metrics(preds["y_true"], preds["y_pred"], proba,
                                 dataset=dataset)
    re_ok = all(int(recomputed[k]) == int(row[k]) for k in ("TN", "FP", "FN", "TP"))
    re_ok &= close(recomputed["accuracy"], (row["TN"] + row["TP"]) / row["n_eval"])
    re_ok &= close(recomputed["cost"], row["cost_pooled"]) if dataset == "credit-g" else True
    ok_all &= check_row(checks, config_id, protocol,
                        "pooled cm/cost == evaluate.py(preds)", bool(re_ok),
                        f"cm={recomputed['confusion_matrix']}")
    if protocol == "holdout":
        counts = preds["y_true"].value_counts().to_dict()
        ok_all &= check_row(
            checks, config_id, protocol, "holdout label counts == manifest",
            counts == manifest[dataset]["test_label_counts"], str(counts))

    return {"config_id": config_id, "dataset": dataset, "model": model,
            "strategy": strategy, "protocol": protocol,
            "pos_label": row["pos_label"], "n_folds": int(row["n_folds"]),
            "n_eval": int(row["n_eval"]),
            **{f"{m}_{s}": row[f"{m}_{s}"] for m in METRICS for s in ("mean", "std")},
            "TN": int(row["TN"]), "FP": int(row["FP"]),
            "FN": int(row["FN"]), "TP": int(row["TP"]),
            "cost_pooled": row["cost_pooled"],
            "runtime_sec": row["runtime_sec"], "audit": "PASS" if ok_all else "FAIL"}


def add_ranks(table):
    """Rank the 4 models within each dataset x protocol; flag RQ2 divergence."""
    models = table[table["strategy"] == "plain"]
    for metric in ("accuracy", "recall", "f1", "auc"):
        rank = (models.groupby(["dataset", "protocol"])[f"{metric}_mean"]
                .rank(ascending=False, method="min"))
        table.loc[models.index, f"{metric}_rank"] = rank
    table[["accuracy_rank", "recall_rank", "f1_rank", "auc_rank"]] = (
        table[["accuracy_rank", "recall_rank", "f1_rank", "auc_rank"]].astype("Int64"))
    plain_rows = table["strategy"] == "plain"
    table.loc[plain_rows, "rank_divergence"] = np.where(
        (table.loc[plain_rows, "accuracy_rank"] != table.loc[plain_rows, "f1_rank"])
        | (table.loc[plain_rows, "accuracy_rank"] != table.loc[plain_rows, "auc_rank"]),
        "yes", "no")
    return table


def main():
    manifest = json.loads((RESULTS / "split_manifest.json").read_text(encoding="utf-8"))
    raw_labels = {
        dataset: pd.read_csv(ROOT / "data" / "raw" / f"{dataset}.csv",
                             na_values="?")[spec["target"]]
        for dataset, spec in DATASETS.items()}

    checks, rows = [], []
    for config_id, dataset, model, strategy in CONFIGS:
        for protocol in PROTOCOLS:
            rows.append(audit_config(config_id, dataset, model, strategy, protocol,
                                     raw_labels, manifest, checks))
    checks_df = pd.DataFrame(checks)
    table = add_ranks(pd.DataFrame(rows))

    out_table = RESULTS / "plain_metrics_v1.csv"
    out_checks = RESULTS / "d3_direction_check_v1.csv"
    table.to_csv(out_table, index=False)
    checks_df.to_csv(out_checks, index=False)

    n_fail = int((checks_df["result"] == "FAIL").sum())
    print(f"checks: {len(checks_df)} total, {n_fail} FAIL")
    print(f"saved: {out_table.relative_to(ROOT)} ({len(table)} rows)")
    print(f"saved: {out_checks.relative_to(ROOT)} ({len(checks_df)} rows)")
    if n_fail:
        print(checks_df[checks_df["result"] == "FAIL"].to_string(index=False))

    # RQ2: ranking divergence, printed for the D3 record
    for (dataset, protocol), grp in table[table["strategy"] == "plain"].groupby(
            ["dataset", "protocol"]):
        print(f"\n[{dataset} / {protocol}] ranking (1 = best)")
        cols = ["model", "accuracy_mean", "accuracy_rank", "recall_mean",
                "recall_rank", "f1_mean", "f1_rank", "auc_mean", "auc_rank",
                "rank_divergence"]
        print(grp[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nZeroR reference rows (accuracy trap for RQ2):")
    zero = table[table["strategy"] == "zero_ref"][
        ["dataset", "protocol", "accuracy_mean", "recall_mean", "f1_mean",
         "auc_mean", "TN", "FP", "FN", "TP"]]
    print(zero.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
