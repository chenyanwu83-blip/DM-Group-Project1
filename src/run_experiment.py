"""
跑实验的主入口。流程对每个配置都一样: 读锁定切分 -> 每折新建一条流水线
-> fit/predict -> 指标全部调 D 的 evaluate.py, 按 D 定的命名写进 results/。

在仓库根目录运行:
    python -m src.run_experiment --baseline            # plain 8 配置全跑 + ZeroR 参照
    python -m src.run_experiment --matrix --strategy threshold
    python -m src.run_experiment --dataset creditg --model lr --protocol cv10   # 跑单个试试

几条不能破的规矩(破了和大家的数字就对不上了):
    - 切分文件(cv_folds_*.csv / *_idx_*.csv)只读, 绝不重新划分;
    - 指标只调 evaluate.py, 不自己另写公式;
    - 输出文件 cv10 一个名字, holdout 加 _holdout 中缀, 不然会互相覆盖;
    - 列严格跟 results/ 里的模板, 一列不多一列不少;
    - 每次的预测明细存 results/preds/, D 要抽查的;
    - records 列表只放同一个 config × protocol, 千万别混着装。
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
sys.path.insert(0, str(ROOT))   # 让 src.* 能被导入

from src.evaluate import compute_metrics, aggregate_folds, FOLD_COLUMNS
from src.c_config import MODELS, DATASETS, RUN_ID, SEED
from src.c_pipelines import load_dataset, build_pipe

SUMMARY_COLUMNS = list(pd.read_csv(ROOT / "results" / "template.csv",
                                   nrows=0).columns)   # 直接抄模板的列, 不手写


def sha_of(dataset_d):
    """sha 直接读 manifest 里记好的, 不自己再算一遍。"""
    manifest = json.loads((ROOT / "results" / "split_manifest.json").read_text(encoding="utf-8"))
    return manifest[dataset_d]["sha256"]


def folds_table(short):
    # 命名有点不统一: 折表用 credit-g(带连字符), 索引文件用 creditg
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
    # 检查折表没有错位: 必须恰好覆盖每一行, 折号 1 到 10
    assert folds["row_idx"].tolist() == list(range(len(X))), f"{short}: 折表行位错位"
    assert sorted(folds["fold_id"].unique()) == list(range(1, 11)), f"{short}: 折号非 1..10"
    sha = sha_of(meta["dataset_d"])
    records, fold_rows, pred_parts = [], [], []
    start = perf_counter()
    for fold_id in range(1, 11):
        test_idx = folds.loc[folds["fold_id"] == fold_id, "row_idx"].to_numpy()
        train_idx = folds.loc[folds["fold_id"] != fold_id, "row_idx"].to_numpy()
        est = make_estimator()   # 每折都是全新实例, 折内自己拟合
        fold_start = perf_counter()
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = est.predict(X.iloc[test_idx])
        prob = est.predict_proba(X.iloc[test_idx])
        runtime = perf_counter() - fold_start
        rec = compute_metrics(y.iloc[test_idx], pred, prob,
                              dataset=meta["dataset_d"], classes=est.classes_)
        records.append(rec)
        row = _meta_row(rec, config_id, model_display, strategy,
                        "cv10", fold_id, sha, runtime)
        row["dataset"] = meta["dataset_d"]
        if hasattr(est, "best_tau_"):   # threshold 策略: 把锁定的 tau 记进 notes 留档
            row["notes"] = f"tau={est.best_tau_:.2f};oof_f1={est.best_oof_f1_:.4f}" + \
                (f";{row['notes']}" if row.get("notes") else "")
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
    est = make_estimator()   # holdout 就拟合一次评估一次
    start = perf_counter()
    est.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = est.predict(X.iloc[hold_idx])
    prob = est.predict_proba(X.iloc[hold_idx])
    runtime = perf_counter() - start
    rec = compute_metrics(y.iloc[hold_idx], pred, prob,
                          dataset=meta["dataset_d"], classes=est.classes_)
    row = _meta_row(rec, config_id, model_display, strategy, "holdout", 1, sha, runtime)
    row["dataset"] = meta["dataset_d"]
    if hasattr(est, "best_tau_"):    # 同上, holdout 也把 tau 记下来
        row["notes"] = f"tau={est.best_tau_:.2f};oof_f1={est.best_oof_f1_:.4f}" + \
            (f";{row['notes']}" if row.get("notes") else "")
    summary = aggregate_folds([rec], config_id=config_id, dataset=meta["dataset_d"],
                              model=model_display, strategy=strategy, protocol="holdout",
                              seed=SEED, run_id=RUN_ID, data_sha256=sha,
                              runtime_sec=runtime)   # std 留 NaN, 不拿 0 冒充
    j = list(est.classes_).index(meta["pos_label"])
    preds = pd.DataFrame({"row_idx": hold_idx, "y_true": y.iloc[hold_idx].to_numpy(),
                          "y_pred": pred, "proba_pos": prob[:, j]})
    save_outputs(config_id, "holdout",
                 [{key: row.get(key) for key in FOLD_COLUMNS}], summary, preds)
    return summary


def one_config(short, model_key, strategy="plain"):
    cfg = f"{short}_{model_key.lower()}_{strategy}_v1"   # 文件名长这样: creditg_dt_plain_v1
    disp = MODELS[model_key]["display"]
    mk = lambda: build_pipe(short, model_key, strategy)
    run_cv10(short, mk, cfg, disp, strategy)
    run_holdout(short, mk, cfg, disp, strategy)


def zero_r(short):
    cfg = f"{short}_zero_ref_v1"   # ZeroR 参照, 顺便当评估链路的试金石
    mk = lambda: DummyClassifier(strategy="most_frequent")
    run_cv10(short, mk, cfg, "ZeroR")
    run_holdout(short, mk, cfg, "ZeroR")


def main():
    ap = argparse.ArgumentParser(description="C3 plain baseline runner (D2 interface)")
    ap.add_argument("--baseline", action="store_true",
                    help="C3 全量: plain 8配置×{cv10,holdout} + ZeroR×2")
    ap.add_argument("--matrix", action="store_true",
                    help="C4 策略全量矩阵: 8配置×{cv10,holdout}, 策略由 --strategy 决定")
    ap.add_argument("--zero", action="store_true", help="只跑 ZeroR 参照")
    ap.add_argument("--dataset", choices=list(DATASETS))
    ap.add_argument("--model", choices=[k.lower() for k in MODELS])
    ap.add_argument("--strategy", choices=["plain", "smote", "cost", "threshold"],
                    default="plain")
    ap.add_argument("--protocol", choices=["cv10", "holdout", "both"], default="both")
    args = ap.parse_args()
    if args.baseline:
        for short in DATASETS:
            for key in MODELS:
                one_config(short, key)
        for short in DATASETS:
            zero_r(short)
    elif args.matrix:
        for short in DATASETS:
            for key in MODELS:
                one_config(short, key, strategy=args.strategy)
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
