"""CDS535 D4: full metrics table v1 (32 configs x 2 protocols) + ensemble add-on table.

Input  : frozen main-matrix results (4 models x 4 strategies x 2 datasets + 2 ZeroR
         refs, cv10 + holdout) and exploratory ensemble results
         (results/ensemble/, bag/ada x plain/smote x 2 datasets x 2 protocols),
         D2 split assets, D2 evaluate.py templates.
Output : results/all_metrics_v1.csv       (D4 deliverable 1, main matrix only)
         results/ensemble_metrics_v1.csv  (D4 deliverable 2, ensemble, kept
                                           separate from the main table)
         results/d4_audit_check_v1.csv    (per-check PASS/FAIL record)

Run from repository root:
    python src/d4_all_metrics.py

No model is trained here. Every number is read from the frozen summary CSVs and
independently re-derived from the archived fold-level CSVs (双侧复算, ddof=0);
tau values are parsed out of the folds `notes` column (the single source of
truth, per D3 §5.2 the summary keeps no tau copy).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluate import (  # noqa: E402
    DATASETS, METRICS, SUMMARY_COLUMNS, FOLD_COLUMNS,
)

RESULTS = ROOT / "results"
PREDS = RESULTS / "preds"
ENS_PREDS = RESULTS / "ensemble" / "preds"
MODELS = ("dt", "rf", "lr", "svc")
MODEL_NAMES = {"dt": "DecisionTree", "rf": "RandomForest",
               "lr": "Logistic", "svc": "SVC"}
STRATEGIES = ("plain", "smote", "cost", "threshold")
PROTOCOLS = ("cv10", "holdout")
HOLDOUT_IDX = {"credit-g": "holdout_idx_creditg.csv", "sick": "holdout_idx_sick.csv"}
TOL = 1e-9
TAU_RE = re.compile(r"tau=([0-9.]+)")

# RQ2 flag rule (documented in docs/D4): high accuracy while minority recall
# collapses. ZeroR rows get their own label (same numeric rule, reference rows).
RQ2_ACC_MIN = 0.70
RQ2_RECALL_MAX = 0.40


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


def collect_unit(config_id, dataset, protocol, manifest, checks, subdir=""):
    """Audit one config x protocol unit; return its row for the metrics table."""
    spec = DATASETS[dataset]
    n_rows = manifest[dataset]["n_rows"]
    suffix = "" if protocol == "cv10" else "_holdout"
    base = RESULTS / subdir
    summary = pd.read_csv(base / f"{config_id}_run1{suffix}_summary.csv")
    folds = pd.read_csv(base / f"{config_id}_run1{suffix}_folds.csv")
    row = summary.iloc[0]
    ok_all = True

    # --- A. schema / meta (D2 template + D1 mapping + frozen split) ---------
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
    ok_all &= check_row(
        checks, config_id, protocol, "n_folds rows on disk",
        len(folds) == (10 if protocol == "cv10" else 1), f"{len(folds)}")
    ok_all &= check_row(checks, config_id, protocol, "protocol field",
                        row["protocol"] == protocol)

    # --- B. folds internal consistency + aggregation re-derivation ----------
    ok_all &= check_row(checks, config_id, protocol, "fold cm sums == n_eval",
                        bool((folds["TN"] + folds["FP"] + folds["FN"] + folds["TP"]
                              == folds["n_eval"]).all()))
    if dataset == "credit-g":
        ok_all &= check_row(
            checks, config_id, protocol, "fold cost == (5FN+FP)/N",
            bool(np.allclose(folds["cost"],
                             (5 * folds["FN"] + folds["FP"]) / folds["n_eval"])))

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
                        "summary mean/std(ddof=0)/n_valid == re-derived from folds",
                        agg_ok, ";".join(agg_bad))
    cm_sum = folds[["TN", "FP", "FN", "TP"]].sum()
    ok_all &= check_row(checks, config_id, protocol, "pooled cm == sum of folds",
                        all(int(cm_sum[k]) == int(row[k])
                            for k in ("TN", "FP", "FN", "TP")),
                        f"TN={cm_sum['TN']},FP={cm_sum['FP']},"
                        f"FN={cm_sum['FN']},TP={cm_sum['TP']}")
    if protocol == "holdout":
        ok_all &= check_row(
            checks, config_id, protocol, "holdout std blank",
            all(pd.isna(row[f"{m}_std"]) for m in METRICS
                if not (m == "cost" and dataset == "sick")))

    # --- C. preds coverage (row counts vs locked splits; label direction was
    #        audited in D3 for plain and C10 for ensemble, not repeated here) --
    if subdir:
        pred_path = ENS_PREDS / f"{config_id}_run1_{protocol}_preds.csv"
    else:
        pred_path = PREDS / f"{config_id}_run1_{protocol}_preds.csv"
    n_pred = len(pd.read_csv(pred_path))
    expected = (n_rows if protocol == "cv10" else len(
        pd.read_csv(RESULTS / HOLDOUT_IDX[dataset])))
    ok_all &= check_row(checks, config_id, protocol, "preds row count == locked coverage",
                        n_pred == expected, f"{n_pred} vs {expected}")

    # --- D. tau parsing (threshold only; folds notes is the single source) ---
    tau_min = tau_max = np.nan
    if row["strategy"] == "threshold":
        taus = folds["notes"].astype(str).map(
            lambda s: (m := TAU_RE.search(s)) and float(m.group(1)))
        taus = pd.Series([t for t in taus if t is not None], dtype=float)
        ok_all &= check_row(checks, config_id, protocol, "tau present in every fold note",
                            len(taus) == len(folds), f"parsed {len(taus)}/{len(folds)}")
        if len(taus) == len(folds):
            tau_min, tau_max = float(taus.min()), float(taus.max())

    return {"config_id": config_id, "dataset": dataset,
            "model": row["model"], "strategy": row["strategy"], "protocol": protocol,
            "pos_label": row["pos_label"], "n_folds": int(row["n_folds"]),
            "n_eval": int(row["n_eval"]),
            **{f"{m}_{s}": row[f"{m}_{s}"] for m in METRICS for s in ("mean", "std")},
            "TN": int(row["TN"]), "FP": int(row["FP"]),
            "FN": int(row["FN"]), "TP": int(row["TP"]),
            "cost_pooled": row["cost_pooled"],
            "tau_min": tau_min, "tau_max": tau_max,
            "runtime_sec": row["runtime_sec"],
            "audit": "PASS" if ok_all else "FAIL"}


def flag_rq2(table):
    """Mark 'accuracy high but minority recall collapses' units (RQ2)."""
    acc = table["accuracy_mean"] >= RQ2_ACC_MIN
    rec = table["recall_mean"] <= RQ2_RECALL_MAX
    table["rq2_flag"] = np.where(
        acc & rec,
        # ZeroR reference rows carry strategy="plain" in the summaries; the
        # model column is the reliable marker.
        np.where(table["model"] == "ZeroR", "zero_ref_trap",
                 "acc_high_recall_low"),
        "")
    return table


def main():
    manifest = json.loads((RESULTS / "split_manifest.json").read_text(encoding="utf-8"))
    ds_keys = {"credit-g": "creditg", "sick": "sick"}

    main_configs = [(f"{ds_keys[d]}_{m}_{s}_v1", d)
                    for d in DATASETS for m in MODELS for s in STRATEGIES]
    main_configs += [(f"{ds_keys[d]}_zero_ref_v1", d) for d in DATASETS]
    ens_configs = [(f"{ds_keys[d]}_{m}_{s}_v1", d)
                   for d in DATASETS for m in ("bag", "ada") for s in ("plain", "smote")]

    checks, main_rows, ens_rows = [], [], []
    for config_id, dataset in main_configs:
        for protocol in PROTOCOLS:
            main_rows.append(collect_unit(config_id, dataset, protocol,
                                          manifest, checks))
    for config_id, dataset in ens_configs:
        for protocol in PROTOCOLS:
            ens_rows.append(collect_unit(config_id, dataset, protocol,
                                         manifest, checks, subdir="ensemble"))

    checks_df = pd.DataFrame(checks)
    main_table = flag_rq2(pd.DataFrame(main_rows))
    ens_table = pd.DataFrame(ens_rows)

    out_main = RESULTS / "all_metrics_v1.csv"
    out_ens = RESULTS / "ensemble_metrics_v1.csv"
    out_checks = RESULTS / "d4_audit_check_v1.csv"
    main_table.to_csv(out_main, index=False)
    ens_table.to_csv(out_ens, index=False)
    checks_df.to_csv(out_checks, index=False)

    n_fail = int((checks_df["result"] == "FAIL").sum())
    print(f"checks: {len(checks_df)} total, {n_fail} FAIL")
    print(f"saved: {out_main.relative_to(ROOT)} ({len(main_table)} rows)")
    print(f"saved: {out_ens.relative_to(ROOT)} ({len(ens_table)} rows)")
    print(f"saved: {out_checks.relative_to(ROOT)} ({len(checks_df)} rows)")
    if n_fail:
        print(checks_df[checks_df["result"] == "FAIL"].to_string(index=False))

    # headline pivots for the D4 record (mean over 10 folds; holdout single)
    for (dataset, protocol), grp in main_table.groupby(["dataset", "protocol"],
                                                       sort=False):
        print(f"\n[{dataset} / {protocol}] f1 mean by model x strategy")
        piv = grp.pivot_table(index="model", columns="strategy", values="f1_mean")
        print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\nRQ2 flagged units (acc>=0.70 & recall<=0.40):")
    cols = ["config_id", "protocol", "accuracy_mean", "recall_mean", "rq2_flag"]
    print(main_table[main_table["rq2_flag"] != ""][cols].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nensemble f1 mean (winner-flip check):")
    print(ens_table.pivot_table(index=["dataset", "protocol"], columns=["model", "strategy"],
                                values="f1_mean").to_string(
        float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
