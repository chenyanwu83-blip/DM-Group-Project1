# -*- coding: utf-8 -*-
"""C3: plain 基线矩阵统一实验入口(D2 接口版)。

用法(仓库根目录):
    python -m src.run_experiment --baseline                 # C3 全量: 8配置(4模型×2数据集)×2协议 + ZeroR 参照
    python -m src.run_experiment --dataset creditg --model lr --protocol cv10   # 单配置冒烟
    python -m src.run_experiment --zero                    # 只跑 ZeroR 参照

硬性约定(D1/D2):
    - 只消费库内锁定切分(cv_folds_*.csv / *_idx_*.csv), 绝不重新划分;
    - 指标一律调 src/evaluate.py 的 compute_metrics/aggregate_folds, 不自写公式;
    - 输出 results/{config_id}_{run_id}_folds/_summary.csv (cv10, D2 原文命名);
      holdout 同名加 _holdout 中缀(config_id 不含协议, 必须以此区分, 防覆盖);
    - 列严格等于模板(FOLD_COLUMNS 28 列 / template.csv 46 列), 模板永远空表头;
    - 预测/正类概率/行位留档 results/preds/ 供 D3 核查;
    - 一个 records 列表只装同一 config × dataset × protocol, 绝不混装(D2 明令)。
"""
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # 保证 src.* 可导入

from src.evaluate import compute_metrics, aggregate_folds, FOLD_COLUMNS
from src.c_config import MODELS, DATASETS, RUN_ID, SEED
from src.c_pipelines import load_dataset, build_pipe

SUMMARY_COLUMNS = list(pd.read_csv(ROOT / "results" / "template.csv",
                                   nrows=0).columns)   # 46 列, 与模板零偏差


def sha_of(dataset_d):
    """data_sha256 取 split_manifest.json 现值, 不自行重算。"""
    manifest = json.loads((ROOT / "results" / "split_manifest.json").read_text(encoding="utf-8"))
    return manifest[dataset_d]["sha256"]


def folds_table(short):
    # 命名不对称: cv 折表用连字符数据集名(credit-g), 索引文件用无连字符后缀(creditg)
    return pd.read_csv(ROOT / "results" / f"cv_folds_{DATASETS[short]['dataset_d']}.csv")


def idx_table(kind, short):
    suffix = "creditg" if short == "creditg" else "sick"
    return pd.read_csv(ROOT / "results" / f"{kind}_idx_{suffix}.csv")["row_idx"].to_numpy()


def save_outputs(config_id, protocol, fold_rows, summary, pred_frame):
    out = ROOT / "results"
    (out / "preds").mkdir(exist_ok=True)
    sfx = "" if protocol == "cv10" else "_holdout"
    pd.DataFrame(fold_rows, columns=FOLD_COLUMNS).to_csv(
        out / f"{config_id}_{RUN_ID}{sfx}_folds.csv", index=False)
    pd.DataFrame([summary]).reindex(columns=SUMMARY_COLUMNS).to_csv(
        out / f"{config_id}_{RUN_ID}{sfx}_summary.csv", index=False)
    pred_frame.to_csv(
        out / "preds" / f"{config_id}_{RUN_ID}_{protocol}_preds.csv", index=False)
    print(f"saved: {config_id} [{protocol}]", flush=True)


def _meta_row(rec, config_id, model_display, strategy, protocol, fold_id, sha, runtime):
    """compute_metrics 结果 + 定位列 → 折行原始字典(再按 FOLD_COLUMNS 取齐)。"""
    row = {"config_id": config_id, "run_id": RUN_ID,
           "model": model_display, "strategy": strategy, "protocol": protocol,
           "fold_id": fold_id, "seed": SEED, "data_sha256": sha,
           "runtime_sec": runtime}
    row.update(rec)
    return row


def run_cv10(short, make_estimator, config_id, model_display, strategy="plain"):
    meta = DATASETS[short]
    X, y = load_dataset(short)
    folds = folds_table(short)
    # D2 防错位断言: 折表必须恰好覆盖全数据每一行, 折号 1..10
    assert folds["row_idx"].tolist() == list(range(len(X))), f"{short}: 折表行位错位"
    assert sorted(folds["fold_id"].unique()) == list(range(1, 11)), f"{short}: 折号非 1..10"
    sha = sha_of(meta["dataset_d"])
    records, fold_rows, pred_parts = [], [], []
    start = perf_counter()
    for fold_id in range(1, 11):
        test_idx = folds.loc[folds["fold_id"] == fold_id, "row_idx"].to_numpy()
        train_idx = folds.loc[folds["fold_id"] != fold_id, "row_idx"].to_numpy()
        est = make_estimator()                    # 每折全新实例: 折内拟合, 无跨折状态
        fold_start = perf_counter()
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = est.predict(X.iloc[test_idx])
        prob = est.predict_proba(X.iloc[test_idx])
        runtime = perf_counter() - fold_start
        rec = compute_metrics(y.iloc[test_idx], pred, prob,
                              dataset=meta["dataset_d"], classes=est.classes_)
        records.append(rec)                       # 只装本配置本协议(D2)
        row = _meta_row(rec, config_id, model_display, strategy,
                        "cv10", fold_id, sha, runtime)
        row["dataset"] = meta["dataset_d"]
        fold_rows.append({key: row.get(key) for key in FOLD_COLUMNS})
        j = list(est.classes_).index(meta["pos_label"])
        pred_parts.append(pd.DataFrame({
            "row_idx": test_idx, "y_true": y.iloc[test_idx].to_numpy(),
            "y_pred": pred, "proba_pos": prob[:, j]}))
    summary = aggregate_folds(records, config_id=config_id, dataset=meta["dataset_d"],
                              model=model_display, strategy=strategy, protocol="cv10",
                              seed=SEED, run_id=RUN_ID, data_sha256=sha,
                              runtime_sec=perf_counter() - start)
    save_outputs(config_id, "cv10", fold_rows, summary,
                 pd.concat(pred_parts, ignore_index=True))
    return summary


def run_holdout(short, make_estimator, config_id, model_display, strategy="plain"):
    meta = DATASETS[short]
    X, y = load_dataset(short)
    train_idx, hold_idx = idx_table("train", short), idx_table("holdout", short)
    assert len(train_idx) + len(hold_idx) == len(X), f"{short}: 训练+留出 != 总行数"
    assert set(train_idx).isdisjoint(hold_idx), f"{short}: 训练/留出索引重叠"
    assert sorted(list(train_idx) + list(hold_idx)) == list(range(len(X))), \
        f"{short}: 索引未恰好覆盖全数据"
    sha = sha_of(meta["dataset_d"])
    est = make_estimator()                        # 锁定配置: fit 一次, 评一次
    start = perf_counter()
    est.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = est.predict(X.iloc[hold_idx])
    prob = est.predict_proba(X.iloc[hold_idx])
    runtime = perf_counter() - start
    rec = compute_metrics(y.iloc[hold_idx], pred, prob,
                          dataset=meta["dataset_d"], classes=est.classes_)
    row = _meta_row(rec, config_id, model_display, strategy, "holdout", 1, sha, runtime)
    row["dataset"] = meta["dataset_d"]
    summary = aggregate_folds([rec], config_id=config_id, dataset=meta["dataset_d"],
                              model=model_display, strategy=strategy, protocol="holdout",
                              seed=SEED, run_id=RUN_ID, data_sha256=sha,
                              runtime_sec=runtime)   # std 列自动 NaN, 不冒充 0
    j = list(est.classes_).index(meta["pos_label"])
    preds = pd.DataFrame({"row_idx": hold_idx, "y_true": y.iloc[hold_idx].to_numpy(),
                          "y_pred": pred, "proba_pos": prob[:, j]})
    save_outputs(config_id, "holdout",
                 [{key: row.get(key) for key in FOLD_COLUMNS}], summary, preds)
    return summary


def one_config(short, model_key, strategy="plain"):
    cfg = f"{short}_{model_key.lower()}_{strategy}_v1"      # D2 命名: creditg_dt_plain_v1
    disp = MODELS[model_key]["display"]
    mk = lambda: build_pipe(short, model_key, strategy)
    run_cv10(short, mk, cfg, disp, strategy)
    run_holdout(short, mk, cfg, disp, strategy)


def zero_r(short):
    cfg = f"{short}_zero_ref_v1"                            # ZeroR 参照(评估链路试金石)
    mk = lambda: DummyClassifier(strategy="most_frequent")
    run_cv10(short, mk, cfg, "ZeroR")
    run_holdout(short, mk, cfg, "ZeroR")


def main():
    ap = argparse.ArgumentParser(description="C3 plain baseline runner (D2 interface)")
    ap.add_argument("--baseline", action="store_true",
                    help="C3 全量: plain 8配置×{cv10,holdout} + ZeroR×2")
    ap.add_argument("--zero", action="store_true", help="只跑 ZeroR 参照")
    ap.add_argument("--dataset", choices=list(DATASETS))
    ap.add_argument("--model", choices=[k.lower() for k in MODELS])
    ap.add_argument("--strategy", choices=["plain", "smote", "cost"], default="plain")
    ap.add_argument("--protocol", choices=["cv10", "holdout", "both"], default="both")
    args = ap.parse_args()
    if args.baseline:
        for short in DATASETS:
            for key in MODELS:
                one_config(short, key)
        for short in DATASETS:
            zero_r(short)
    elif args.zero:
        for short in DATASETS:
            zero_r(short)
    elif args.dataset and args.model:
        key = next(k for k in MODELS if k.lower() == args.model)
        cfg = f"{args.dataset}_{key.lower()}_{args.strategy}_v1"
        disp, strategy = MODELS[key]["display"], args.strategy
        mk = lambda: build_pipe(args.dataset, key, strategy)
        if args.protocol in ("cv10", "both"):
            run_cv10(args.dataset, mk, cfg, disp, strategy)
        if args.protocol in ("holdout", "both"):
            run_holdout(args.dataset, mk, cfg, disp, strategy)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
