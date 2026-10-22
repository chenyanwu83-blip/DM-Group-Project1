"""
B · 集成增强研究 · sick 侧实验入口（V4 方案 · 2026-10-21 任务）
================================================================
依据《分工与时间线执行方案 V4.0 · 集成增强版》§1.3（超参一次锁定）：
  模型   BaggingClassifier(DT) / AdaBoostClassifier(DT)
         基学习器 = 主矩阵同款 DecisionTree(criterion='entropy',
         ccp_alpha=0.01, random_state=42)；n_estimators=100（对齐 RF）
  策略   plain / smote（SMOTE 在 imblearn Pipeline 内，仅训练折）
  协议   cv10 + holdout —— 复用 D 的锁定切分，绝不重新划分
  指标   统一走 src/evaluate.py；列与 results/template.csv 零偏差
  产出   results/ensemble/{ds}_{bag,ada}_{plain,smote}_v1_run1_*.csv
         + results/ensemble/preds/（独立命名空间，不碰主矩阵）

用法（VS-Code 目录 + cds535 环境）：
    python src/b7x_ensemble_study.py --dataset sick --model bag --strategy plain --protocol cv10  # 冒烟
    python src/b7x_ensemble_study.py --dataset sick --all                                          # 4 配置全量
    python src/b7x_ensemble_study.py --dataset creditg --all    # A 镜像 credit-g 侧时用
"""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.c_config import DATASETS, RUN_ID, SEED
from src.c_pipelines import load_dataset, make_pre
from src.evaluate import FOLD_COLUMNS, aggregate_folds, compute_metrics

OUT_DIR = ROOT / "results" / "ensemble"
SUMMARY_COLUMNS = list(pd.read_csv(ROOT / "results" / "template.csv",
                                   nrows=0).columns)   # 与 D 模板零偏差


def base_tree():
    """基学习器 = 主矩阵同款 DT（c_config MODELS['DT'] 的逐字超参）。"""
    return DecisionTreeClassifier(criterion="entropy", ccp_alpha=0.01,
                                  random_state=SEED)


def make_ens(model_key):
    if model_key == "bag":
        return BaggingClassifier(estimator=base_tree(), n_estimators=100,
                                 random_state=SEED, n_jobs=-1)
    return AdaBoostClassifier(estimator=base_tree(), n_estimators=100,
                              random_state=SEED)


def build_ens_pipe(short, model_key, strategy):
    """组装：预处理(树通道) → (SMOTE) → 集成分类器。每折新建，折内拟合。"""
    steps = [("pre", make_pre(short, linear=False))]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("clf", make_ens(model_key)))
    return Pipeline(steps)


def folds_table(short):
    # 命名不对称沿用 C 的 run_experiment：折表用连字符名，索引用无连字符后缀
    return pd.read_csv(ROOT / "results" / f"cv_folds_{DATASETS[short]['dataset_d']}.csv")


def idx_table(kind, short):
    suffix = "creditg" if short == "creditg" else "sick"
    return pd.read_csv(ROOT / "results" / f"{kind}_idx_{suffix}.csv")["row_idx"].to_numpy()


def sha_of(short):
    manifest = json.loads((ROOT / "results" / "split_manifest.json").read_text(encoding="utf-8"))
    return manifest[DATASETS[short]["dataset_d"]]["sha256"]


def save_outputs(config_id, protocol, fold_rows, summary, pred_frame):
    (OUT_DIR / "preds").mkdir(parents=True, exist_ok=True)
    sfx = "" if protocol == "cv10" else "_holdout"
    pd.DataFrame(fold_rows, columns=FOLD_COLUMNS).to_csv(
        OUT_DIR / f"{config_id}_{RUN_ID}{sfx}_folds.csv", index=False)
    pd.DataFrame([summary]).reindex(columns=SUMMARY_COLUMNS).to_csv(
        OUT_DIR / f"{config_id}_{RUN_ID}{sfx}_summary.csv", index=False)
    pred_frame.to_csv(
        OUT_DIR / "preds" / f"{config_id}_{RUN_ID}_{protocol}_preds.csv", index=False)
    print(f"saved: {config_id} [{protocol}]", flush=True)


def _fold_row(rec, config_id, model_key, strategy, protocol, fold_id, sha, runtime, short):
    row = {"config_id": config_id, "run_id": RUN_ID, "model": model_key,
           "strategy": strategy, "protocol": protocol, "fold_id": fold_id,
           "seed": SEED, "data_sha256": sha, "runtime_sec": runtime,
           "dataset": DATASETS[short]["dataset_d"]}
    row.update(rec)
    return {key: row.get(key) for key in FOLD_COLUMNS}


def run_cv10(short, model_key, strategy):
    meta = DATASETS[short]
    config_id = f"{short}_{model_key}_{strategy}_v1"
    X, y = load_dataset(short)
    folds = folds_table(short)
    assert folds["row_idx"].tolist() == list(range(len(X))), f"{short}: 折表行位错位"
    assert sorted(folds["fold_id"].unique()) == list(range(1, 11)), f"{short}: 折号非 1..10"
    sha = sha_of(short)
    records, fold_rows, pred_parts = [], [], []
    start = perf_counter()
    for fold_id in range(1, 11):
        test_idx = folds.loc[folds["fold_id"] == fold_id, "row_idx"].to_numpy()
        train_idx = folds.loc[folds["fold_id"] != fold_id, "row_idx"].to_numpy()
        est = build_ens_pipe(short, model_key, strategy)      # 每折全新：折内拟合
        t0 = perf_counter()
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = est.predict(X.iloc[test_idx])
        prob = est.predict_proba(X.iloc[test_idx])
        runtime = perf_counter() - t0
        rec = compute_metrics(y.iloc[test_idx], pred, prob,
                              dataset=meta["dataset_d"], classes=est.classes_)
        records.append(rec)
        fold_rows.append(_fold_row(rec, config_id, model_key, strategy,
                                   "cv10", fold_id, sha, runtime, short))
        j = list(est.classes_).index(meta["pos_label"])
        pred_parts.append(pd.DataFrame({
            "row_idx": test_idx, "y_true": y.iloc[test_idx].to_numpy(),
            "y_pred": pred, "proba_pos": prob[:, j]}))
    summary = aggregate_folds(records, config_id=config_id,
                              dataset=meta["dataset_d"], model=model_key,
                              strategy=strategy, protocol="cv10", seed=SEED,
                              run_id=RUN_ID, data_sha256=sha,
                              runtime_sec=perf_counter() - start)
    save_outputs(config_id, "cv10", fold_rows, summary,
                 pd.concat(pred_parts, ignore_index=True))
    return summary


def run_holdout(short, model_key, strategy):
    meta = DATASETS[short]
    config_id = f"{short}_{model_key}_{strategy}_v1"
    X, y = load_dataset(short)
    train_idx, hold_idx = idx_table("train", short), idx_table("holdout", short)
    assert len(train_idx) + len(hold_idx) == len(X)
    assert set(train_idx).isdisjoint(hold_idx)
    sha = sha_of(short)
    est = build_ens_pipe(short, model_key, strategy)          # 锁定配置：fit 一次评一次
    start = perf_counter()
    est.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = est.predict(X.iloc[hold_idx])
    prob = est.predict_proba(X.iloc[hold_idx])
    runtime = perf_counter() - start
    rec = compute_metrics(y.iloc[hold_idx], pred, prob,
                          dataset=meta["dataset_d"], classes=est.classes_)
    fold_rows = [_fold_row(rec, config_id, model_key, strategy, "holdout",
                           1, sha, runtime, short)]
    summary = aggregate_folds([rec], config_id=config_id,
                              dataset=meta["dataset_d"], model=model_key,
                              strategy=strategy, protocol="holdout", seed=SEED,
                              run_id=RUN_ID, data_sha256=sha, runtime_sec=runtime)
    j = list(est.classes_).index(meta["pos_label"])
    preds = pd.DataFrame({"row_idx": hold_idx,
                          "y_true": y.iloc[hold_idx].to_numpy(),
                          "y_pred": pred, "proba_pos": prob[:, j]})
    save_outputs(config_id, "holdout", fold_rows, summary, preds)
    return summary


def main():
    ap = argparse.ArgumentParser(description="B: ensemble study runner (V4 plan, D2 interface)")
    ap.add_argument("--dataset", choices=["sick", "creditg"], default="sick")
    ap.add_argument("--model", choices=["bag", "ada"])
    ap.add_argument("--strategy", choices=["plain", "smote"], default="plain")
    ap.add_argument("--protocol", choices=["cv10", "holdout", "both"], default="both")
    ap.add_argument("--all", action="store_true", help="4 配置全量（bag/ada × plain/smote）")
    args = ap.parse_args()
    if args.all:
        for model_key in ("bag", "ada"):
            for strategy in ("plain", "smote"):
                run_cv10(args.dataset, model_key, strategy)
                run_holdout(args.dataset, model_key, strategy)
    elif args.model:
        if args.protocol in ("cv10", "both"):
            run_cv10(args.dataset, args.model, args.strategy)
        if args.protocol in ("holdout", "both"):
            run_holdout(args.dataset, args.model, args.strategy)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
