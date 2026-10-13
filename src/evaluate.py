"""CDS535 D2: binary metrics, fold aggregation, templates and split audit.

Run from repository root:
    python src/evaluate.py
    python src/evaluate.py --check-data --prepare-holdout

C/A/B integration:
    from src.evaluate import compute_metrics, aggregate_folds
    metrics = compute_metrics(
        y_test, y_pred, pipe.predict_proba(X_test),
        pos_label="bad", dataset="credit-g", classes=pipe.classes_)

No model is trained here. Self-test samples are never saved as project results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, cohen_kappa_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

DATASETS = {
    "credit-g": {"target": "class", "positive": "bad", "negative": "good"},
    "sick": {"target": "Class", "positive": "sick", "negative": "negative"},
}
METRICS = (
    "accuracy", "precision", "recall", "f1", "auc",
    "kappa", "balanced_accuracy", "macro_f1", "cost",
)
META_COLUMNS = [
    "config_id", "run_id", "dataset", "model", "strategy", "protocol",
    "n_folds", "seed", "pos_label", "neg_label", "data_sha256",
]
SUMMARY_COLUMNS = META_COLUMNS + [
    col for metric in METRICS for col in
    (f"{metric}_mean", f"{metric}_std", f"{metric}_n_valid")
] + [
    "n_eval", "TN", "FP", "FN", "TP", "confusion_matrix",
    "cost_pooled", "precision_undefined_folds", "recall_undefined_folds",
    "runtime_sec", "notes",
]
FOLD_COLUMNS = [
    "config_id", "run_id", "dataset", "model", "strategy", "protocol",
    "fold_id", "seed", "pos_label", "neg_label", "data_sha256",
] + list(METRICS) + [
    "n_eval", "TN", "FP", "FN", "TP", "precision_undefined",
    "recall_undefined", "runtime_sec", "notes",
]


def _vector(values, name):
    arr = np.asarray(values)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError(f"{name} must be a non-empty 1-D array.")
    if pd.isna(arr).any():
        raise ValueError(f"{name} contains missing labels.")
    return arr


def positive_probability(y_proba, pos_label, classes=None):
    """1-D input MUST already be P(positive); 2-D input requires classes_."""
    arr = np.asarray(y_proba, dtype=float)
    if arr.ndim == 2:
        if classes is None:
            raise ValueError("For 2-D predict_proba, supply classes=pipe.classes_.")
        order = list(classes)
        if len(order) != 2 or arr.shape[1] != len(order):
            raise ValueError("Only binary classification is supported.")
        if len(set(order)) != 2 or pos_label not in order:
            raise ValueError("classes_ must contain two distinct labels including positive.")
        if not np.allclose(arr.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("predict_proba rows must sum to 1.")
        arr = arr[:, order.index(pos_label)]
    elif arr.ndim != 1:
        raise ValueError("Probabilities must be 1-D or 2-D.")
    if not np.isfinite(arr).all() or ((arr < 0) | (arr > 1)).any():
        raise ValueError("Probabilities must be finite and between 0 and 1.")
    return arr


def compute_metrics(y_true, y_pred, y_proba_pos=None, pos_label=None,
                    dataset=None, *, classes=None, neg_label=None):
    """Return all eight metrics, explicit confusion matrix and credit-g cost.

    Inputs keep their label types: no implicit string conversion.
    For project datasets use original string labels. If your estimator uses
    0/1, convert truth, predictions and classes_ back to project labels first.
    Missing probabilities or single-class truth give AUC=NaN with a reason.
    """
    yt, yp = _vector(y_true, "y_true"), _vector(y_pred, "y_pred")
    if len(yt) != len(yp):
        raise ValueError("Truth and prediction lengths differ.")
    if dataset is not None:
        if dataset not in DATASETS:
            raise ValueError("dataset must be 'credit-g' or 'sick'.")
        spec = DATASETS[dataset]
        pos_label = spec["positive"] if pos_label is None else pos_label
        neg_label = spec["negative"] if neg_label is None else neg_label
        if (pos_label, neg_label) != (spec["positive"], spec["negative"]):
            raise ValueError("Project label mapping disagrees with D1.")
    if pos_label is None:
        raise ValueError("Specify pos_label or dataset.")
    observed = set(yt.tolist()) | set(yp.tolist())
    if neg_label is None:
        candidates = observed - {pos_label}
        if len(candidates) != 1:
            raise ValueError("Cannot infer negative label; supply neg_label.")
        neg_label = candidates.pop()
    if pos_label == neg_label or not observed <= {pos_label, neg_label}:
        raise ValueError("Unexpected labels: check target column and mapping.")

    # Map explicitly, so positive may be classes_[0] OR classes_[1].
    tb, pb = (yt == pos_label).astype(int), (yp == pos_label).astype(int)
    cm = confusion_matrix(tb, pb, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in cm.ravel())
    notes = []
    if tp + fp == 0:
        notes.append("Precision denominator is zero; reported as 0.")
    if tp + fn == 0:
        notes.append("Recall denominator is zero; reported as 0.")
    auc = np.nan
    if y_proba_pos is None:
        notes.append("AUC unavailable: no positive probability supplied.")
    else:
        scores = positive_probability(y_proba_pos, pos_label, classes)
        if len(scores) != len(yt):
            raise ValueError("Probability and truth lengths differ.")
        if len(np.unique(tb)) < 2:
            notes.append("AUC unavailable: truth contains only one class.")
        else:
            auc = float(roc_auc_score(tb, scores))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.simplefilter("ignore", UserWarning)
        kappa = float(cohen_kappa_score(tb, pb, labels=[0, 1]))
        # Force binary balanced accuracy, including the single-class edge case.
        if len(np.unique(tb)) == 2:
            balanced = float(balanced_accuracy_score(tb, pb))
        else:
            balanced = np.nan
    if not np.isfinite(kappa):
        notes.append("Kappa unavailable: degenerate class marginals.")
    if not np.isfinite(balanced):
        notes.append("Balanced accuracy unavailable: truth contains one class.")
    return {
        "accuracy": float(accuracy_score(tb, pb)),
        "precision": float(precision_score(tb, pb, zero_division=0)),
        "recall": float(recall_score(tb, pb, zero_division=0)),
        "f1": float(f1_score(tb, pb, zero_division=0)),
        "auc": auc, "kappa": kappa, "balanced_accuracy": balanced,
        "macro_f1": float(f1_score(tb, pb, labels=[0, 1],
                                  average="macro", zero_division=0)),
        "cost": (5 * fn + fp) / len(yt) if dataset == "credit-g" else np.nan,
        "n_eval": len(yt), "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "labels": [neg_label, pos_label], "confusion_matrix": cm.tolist(),
        "pos_label": pos_label, "neg_label": neg_label,
        "precision_undefined": tp + fp == 0,
        "recall_undefined": tp + fn == 0, "notes": "; ".join(notes),
    }


def aggregate_folds(fold_records, *, config_id, dataset, model, strategy,
                    protocol="cv10", seed=42, run_id="run1",
                    data_sha256="", runtime_sec=None):
    """One configuration, one run, one dataset, one protocol per call.

    cv10 requires exactly 10 fold metric dictionaries; holdout requires 1.
    Proposed D1 convention: numpy population SD (ddof=0), matching C1.
    Holdout SD is NaN (blank in CSV), never an invented zero.
    """
    if protocol not in {"cv10", "holdout"} or dataset not in DATASETS:
        raise ValueError("Unknown protocol or dataset.")
    count = 10 if protocol == "cv10" else 1
    if len(fold_records) != count:
        raise ValueError(f"{protocol} requires {count} records.")
    spec = DATASETS[dataset]
    for rec in fold_records:
        if (rec["pos_label"], rec["neg_label"]) != (
                spec["positive"], spec["negative"]):
            raise ValueError("Fold label direction does not match dataset.")
        for key in METRICS:
            if key not in rec:
                raise ValueError(f"Missing metric: {key}")
        if int(rec["n_eval"]) != sum(int(rec[k]) for k in ("TN", "FP", "FN", "TP")):
            raise ValueError("Confusion counts do not sum to n_eval.")
        expected_cost = (5 * rec["FN"] + rec["FP"]) / rec["n_eval"]
        if dataset == "credit-g" and not np.isclose(rec["cost"], expected_cost):
            raise ValueError("credit-g cost disagrees with confusion counts.")
        if dataset == "sick" and np.isfinite(rec["cost"]):
            raise ValueError("sick must not report credit-g cost.")
    out = {
        "config_id": config_id, "run_id": run_id, "dataset": dataset,
        "model": model, "strategy": strategy, "protocol": protocol,
        "n_folds": count, "seed": seed, "pos_label": spec["positive"],
        "neg_label": spec["negative"], "data_sha256": data_sha256,
    }
    for metric in METRICS:
        values = np.array([r[metric] for r in fold_records], dtype=float)
        finite = values[np.isfinite(values)]
        out[f"{metric}_n_valid"] = len(finite)
        out[f"{metric}_mean"] = float(finite.mean()) if len(finite) else np.nan
        out[f"{metric}_std"] = (
            float(finite.std(ddof=0)) if protocol == "cv10" and len(finite) == 10
            else np.nan)
    for name in ("n_eval", "TN", "FP", "FN", "TP"):
        out[name] = sum(int(r[name]) for r in fold_records)
    out["confusion_matrix"] = json.dumps(
        [[out["TN"], out["FP"]], [out["FN"], out["TP"]]])
    out["cost_pooled"] = (
        (5 * out["FN"] + out["FP"]) / out["n_eval"]
        if dataset == "credit-g" else np.nan)
    out["precision_undefined_folds"] = sum(
        int(r["precision_undefined"]) for r in fold_records)
    out["recall_undefined_folds"] = sum(
        int(r["recall_undefined"]) for r in fold_records)
    out["runtime_sec"] = np.nan if runtime_sec is None else runtime_sec
    notes = sorted({r.get("notes", "") for r in fold_records} - {""})
    if any(out[f"{m}_n_valid"] != count for m in METRICS if
           not (m == "cost" and dataset == "sick")):
        notes.append("Incomplete metrics: means use valid folds; SD left blank.")
    out["notes"] = "; ".join(notes)
    return out


def make_template(results_dir="results"):
    """Create header-only templates. Never overwrite an existing template."""
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name, columns in (("template.csv", SUMMARY_COLUMNS),
                          ("fold_template.csv", FOLD_COLUMNS)):
        path = directory / name
        if path.exists():
            if list(pd.read_csv(path, nrows=0).columns) != columns:
                raise ValueError(f"{path} already exists with a different schema.")
            print(f"EXISTS: {path}; unchanged.")
        else:
            pd.DataFrame(columns=columns).to_csv(path, index=False)
            print(f"CREATED: {path} (header only; no experimental numbers).")


def check_and_prepare(root=".", prepare=False):
    """Audit actual raw labels; optionally save position-based split indices.

    Independent full-data CV and 70:30 holdout are separate fixed-config
    protocols. This function saves indices only and never trains a model.
    """
    root = Path(root)
    results = root / "results"
    results.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for dataset, spec in DATASETS.items():
        path = root / "data" / "raw" / f"{dataset}.csv"
        df = pd.read_csv(path, na_values="?")
        target = spec["target"]
        if target not in df or df[target].isna().any():
            raise ValueError(f"{dataset}: target missing or contains missing labels.")
        if set(df[target].unique()) != {spec["positive"], spec["negative"]}:
            raise ValueError(f"{dataset}: unexpected target labels.")
        positions = np.arange(len(df))
        train, test = train_test_split(
            positions, test_size=.3, stratify=df[target], random_state=42)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{dataset}: {df.shape}; target={target}; "
              f"labels={df[target].value_counts().to_dict()}")
        print(f"  holdout train={len(train)} test={len(test)}; "
              f"test labels={df.iloc[test][target].value_counts().to_dict()}")
        print(f"  SHA256={digest}")
        entry = {
            "file": f"data/raw/{dataset}.csv", "sha256": digest,
            "target": target, "positive": spec["positive"],
            "negative": spec["negative"], "seed": 42, "test_size": .3,
            "n_rows": len(df), "n_train": len(train), "n_test": len(test),
            "test_label_counts": df.iloc[test][target].value_counts().to_dict(),
            "index_type": "zero-based original CSV row position, excluding header",
        }
        if prepare:
            candidates = {}
            for prefix, ids in (("train_idx", train), ("holdout_idx", test)):
                # creditg matches the current A-side specification.
                suffix = "creditg" if dataset == "credit-g" else "sick"
                candidates[f"{prefix}_{suffix}.csv"] = pd.DataFrame({"row_idx": ids})
            assignments = np.empty(len(df), dtype=int)
            cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            for fold_id, (_, valid) in enumerate(cv.split(positions, df[target]), 1):
                assignments[valid] = fold_id
            candidates[f"cv_folds_{dataset}.csv"] = pd.DataFrame({
                "row_idx": positions, "fold_id": assignments})
            for filename, table in candidates.items():
                dest = results / filename
                if dest.exists():
                    if not pd.read_csv(dest).equals(table):
                        raise ValueError(f"{dest} differs. Do not overwrite locked splits.")
                    print(f"  EXISTS: {dest}; identical.")
                else:
                    table.to_csv(dest, index=False)
                    print(f"  SAVED: {dest}")
            entry["index_files"] = list(candidates)
        manifest[dataset] = entry
    if prepare:
        dest = results / "split_manifest.json"
        if dest.exists() and json.loads(dest.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing manifest differs; ask C/D before changing data.")
        dest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return manifest


def self_test():
    # Asymmetric confusion matrix detects TN/FP/FN/TP ordering errors.
    yt = ["good", "good", "good", "bad", "bad"]
    yp = ["good", "good", "bad", "good", "bad"]
    prob = [.1, .2, .7, .4, .9]
    m = compute_metrics(yt, yp, prob, dataset="credit-g")
    assert m["confusion_matrix"] == [[2, 1], [1, 1]]
    assert np.isclose(m["cost"], 6 / 5)
    assert np.isclose(m["auc"], 5 / 6)
    p2 = np.column_stack([prob, 1 - np.array(prob)])
    m2 = compute_metrics(yt, yp, p2, dataset="credit-g", classes=["bad", "good"])
    assert np.isclose(m2["auc"], m["auc"])
    m3 = compute_metrics(yt, ["good"] * 5, prob, dataset="credit-g")
    assert m3["precision"] == 0 and m3["precision_undefined"]
    m4 = compute_metrics(["negative", "sick"], ["negative", "sick"],
                         [.1, .9], dataset="sick")
    assert np.isnan(m4["cost"])
    missing = compute_metrics(yt, yp, dataset="credit-g")
    assert np.isnan(missing["auc"]) and "no positive" in missing["notes"]
    single = compute_metrics(["good"] * 2, ["good"] * 2, [.1, .2],
                             dataset="credit-g")
    assert np.isnan(single["auc"]) and single["confusion_matrix"] == [[2, 0], [0, 0]]
    numeric = compute_metrics([0, 0, 1], [0, 1, 1], [.1, .6, .8],
                              pos_label=1, neg_label=0)
    assert numeric["confusion_matrix"] == [[1, 1], [0, 1]]
    holdout = aggregate_folds([m], config_id="TEST", dataset="credit-g",
                              model="TEST", strategy="plain", protocol="holdout")
    assert np.isnan(holdout["accuracy_std"])
    cv = aggregate_folds([m] * 10, config_id="TEST", dataset="credit-g",
                         model="TEST", strategy="plain")
    assert cv["n_folds"] == 10 and cv["TN"] == 20
    assert np.isclose(cv["cost_pooled"], 1.2)
    try:
        compute_metrics(yt, yp, p2, dataset="credit-g")
    except ValueError:
        pass
    else:
        raise AssertionError("2-D probabilities without classes_ must fail.")
    print("SELF-TEST OK: metrics, direction, cost, edge cases and aggregation.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root; run there by default.")
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--prepare-holdout", action="store_true")
    args = parser.parse_args()
    self_test()
    make_template(Path(args.root) / "results")
    if args.check_data or args.prepare_holdout:
        check_and_prepare(args.root, prepare=args.prepare_holdout)


if __name__ == "__main__":
    main()
