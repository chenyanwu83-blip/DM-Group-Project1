# -*- coding: utf-8 -*-
"""C5: 主矩阵折级配对 t 检验 —— Nadeau & Bengio (2003) 方差校正版
（对应 Weka Experimenter 的 corrected resampled t-test）。

用法(仓库根目录):
    python -m src.stats_test                   # 全量: 核验锚点 + 2数据集×4指标×C(16,2)配对 + 关键对照
    python -m src.stats_test --self-test-only  # 只跑核验锚点, 不写任何结果文件

依据:
  - docs/D1_评估协议_v1.0.md   锁定切分/逐折原始表"供后续配对统计"(§D2数据交接规范)
  - docs/D2_给ABC的调用说明.md  folds 文件命名与列口径
  - V3 方案 C5: "src/stats_test.py 用 scipy.stats.t.sf 实现 Nadeau & Bengio (2003)
    校正折级配对 t 检验(对应原 corrected resampled t-test), 出跨模型对比 p 值表"

公式(同折配对, k=10 折):
    d_f = x_f - y_f ;  dbar = mean(d) ;  s2 = var(d, ddof=1)
    Var(dbar) = s2 * (1/k + n_test/n_train)      # NB 校正: 10折时 ≈ 2.11x 朴素方差
    t = dbar / sqrt(Var(dbar)) ;  df = k-1 ;  p = 2 * t.sf(|t|, df)

边界(不造数):
    s2=0 且 dbar=0  → 两配置逐折逐位相同(如 sick 等代价对照): p=1.0, note="identical"
    s2=0 且 dbar!=0 → 常数差但无方差: t 发散, 仅标方向, p 置 NaN 并 note 说明
多重比较: pairs 表同时给 Holm(1979) 族内校正 p(族 = 数据集×指标的全部配对);
    显著性星号按原始 p(α=0.05) 标注, 报告端如何呈现由 D6 按 D1"注明多重比较"口径决定。
跨数据集: 不同样本无法同折配对, 不做跨数据集 t 检验; key 表两侧并列仅供 D6 描述性对比。

指标范围: accuracy / recall / f1 / auc(4 项, 对应 RQ1 策略增益与 RQ2 指标分歧;
kappa/balanced_accuracy/macro_f1 不入检验, 需要时按同框架追加)。
ZeroR 参照(results/{ds}_zero_ref_v1_*)不入配对矩阵(V3: "不计入 4 模型")。

输出(独立命名空间 results/ttest/, 不触碰主矩阵/冻结资产):
    results/ttest/{ds}_ttest_pairs_{metric}_v1.csv    长表: 该数据集该指标全部 C(16,2)=120 配对
    results/ttest/{ds}_ttest_matrix_{metric}_v1.csv   16x16 下三角速览(原始 p + 星号)
    results/ttest/key_comparisons_v1.csv              关键对照: 各模型 smote/cost/threshold vs plain
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                      # 保证 src.* 可导入(与 run_experiment 一致)

from src.c_config import DATASETS, MODELS, RUN_ID, SEED, STRATEGIES

METRICS = ["accuracy", "recall", "f1", "auc"]
KEY_STRATS = ["smote", "cost", "threshold"]        # vs plain 的关键对照
FAMILY_ALPHA = 0.05

MODEL_KEYS = [k.lower() for k in MODELS]           # 文件名口径: dt/rf/lr/svc


# ---------------------------------------------------------------- 核验锚点
def load_folds(short):
    """读 16 配置 folds 表; 顺带做 D2 口径断言(与 run_experiment 同款, 不放宽)。"""
    frames = {}
    for m in MODEL_KEYS:
        for s in STRATEGIES:
            cid = f"{short}_{m}_{s}_v1"
            f = pd.read_csv(ROOT / "results" / f"{cid}_{RUN_ID}_folds.csv")
            assert sorted(f["fold_id"]) == list(range(1, 11)), f"{cid}: fold_id 非 1..10"
            assert (f["protocol"] == "cv10").all() and (f["seed"] == SEED).all(), f"{cid}: 协议/种子漂移"
            frames[cid] = f
    return frames


def self_test(folds_all):
    """锚点核对(打印 + 返回 FAIL 列表); 结果同时写入运行记录的素材。"""
    fails, notes = [], []
    manifest = json.loads((ROOT / "results" / "split_manifest.json").read_text(encoding="utf-8"))
    tmpl_cols = list(pd.read_csv(ROOT / "results" / "fold_template.csv", nrows=0).columns)

    for short, meta in DATASETS.items():
        frames = folds_all[short]
        # ① 结构: 列=模板 / sha=manifest / 指标无 NaN
        for cid, f in frames.items():
            if list(f.columns) != tmpl_cols:
                fails.append(f"{cid}: folds 列 != fold_template")
            if not (f["data_sha256"] == manifest[meta["dataset_d"]]["sha256"]).all():
                fails.append(f"{cid}: data_sha256 != split_manifest")
            if f[METRICS].isna().any().any():
                fails.append(f"{cid}: 检验指标含 NaN")
        # ② 锁定切分: 16 配置 n_eval 逐折一致
        nevals = {tuple(f["n_eval"].tolist()) for f in frames.values()}
        if len(nevals) != 1:
            fails.append(f"{short}: 各配置 n_eval 不一致 → 折未锁定")
        n_test = float(np.mean(list(nevals)[0]))
        n_train = meta["n_rows"] - n_test
        notes.append(f"{short}: 16 配置折大小逐位一致(锁定切分 PASS), "
                     f"n_test(均)={n_test:.1f}, n_train(均)={n_train:.1f}")
        # ③ folds 均值 == summary 均值(ddof=0 口径)
        for cid, f in frames.items():
            su = pd.read_csv(ROOT / "results" / f"{cid}_{RUN_ID}_summary.csv")
            for met in METRICS:
                if abs(f[met].mean() - su[f"{met}_mean"].iloc[0]) > 1e-12:
                    fails.append(f"{cid}: {met} folds 均值 != summary")
                if abs(f[met].std(ddof=0) - su[f"{met}_std"].iloc[0]) > 1e-12:
                    fails.append(f"{cid}: {met} folds std(ddof=0) != summary")
        # ④ C4 锚点复现: threshold 与 plain 的 auc 逐折逐位相等(最终重拟合=同款流水线)
        for m in MODEL_KEYS:
            a = frames[f"{short}_{m}_plain_v1"]["auc"].to_numpy()
            b = frames[f"{short}_{m}_threshold_v1"]["auc"].to_numpy()
            if not (a == b).all():
                fails.append(f"ANCHOR C4 FAIL: {short}_{m} plain auc != threshold auc")
        # ⑤ sick 等代价对照: B5 预期 ≈ plain; 逐位盘点(允许 RF 平票翻转, 如实记录)
        if short == "sick":
            ident = []
            for m in MODEL_KEYS:
                a = frames[f"sick_{m}_plain_v1"][METRICS]
                b = frames[f"sick_{m}_cost_v1"][METRICS]
                if a.equals(b):
                    ident.append(m)
                else:
                    dmax = np.nanmax(np.abs(a.astype(float).to_numpy() - b.astype(float).to_numpy()))
                    notes.append(f"sick_{m}: cost vs plain 非逐位相同(maxΔ={dmax:.4f}), "
                                 f"判定为显式等权 sample_weight 下树分裂平票翻转(RF 常见), 如实入检验")
            notes.append(f"sick 等代价对照逐位相同: {len(ident)}/4 ({','.join(ident) or '无'})")
    print("[self-test] " + " ; ".join(notes))
    print(f"[self-test] 锚点 FAIL 数: {len(fails)}")
    for x in fails:
        print("  FAIL:", x)
    return fails


# ---------------------------------------------------------------- 检验本体
def nb_paired_t(x, y, n_test, n_train):
    """Nadeau & Bengio (2003) 校正折级配对 t 检验。返回 (dbar, t, p, note)。"""
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    k = len(d)
    dbar = float(d.mean())
    s2 = float(d.var(ddof=1))
    if s2 == 0.0:
        if dbar == 0.0:
            return dbar, float("nan"), 1.0, "identical"          # 逐位相同: 无差异, p=1
        return dbar, float("inf") * np.sign(dbar), float("nan"), \
            "zero_variance_const_diff"                            # 常数差: 方差 0, t 发散, 不造 p
    var_dbar = s2 * (1.0 / k + n_test / n_train)                  # NB 校正方差
    t = dbar / np.sqrt(var_dbar)
    p = float(2.0 * stats.t.sf(abs(t), df=k - 1))
    return dbar, float(t), p, ""


def sig_mark(p):
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def holm(pvals):
    """Holm(1979) step-down 校正, 族内执行; NaN 原样保留不参与排序。"""
    p = np.asarray(pvals, dtype=float)
    m = int(np.isfinite(p).sum())
    out = np.full_like(p, np.nan)
    idx = np.argsort(np.where(np.isfinite(p), p, np.inf))[:m]
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def run():
    folds_all = {short: load_folds(short) for short in DATASETS}
    if self_test(folds_all):
        print("[abort] 锚点未过, 不产出检验结果(不基于可疑数据造表)")
        sys.exit(1)

    out_dir = ROOT / "results" / "ttest"
    out_dir.mkdir(exist_ok=True)
    key_rows = []

    for short, meta in DATASETS.items():
        frames = folds_all[short]
        n_test = float(np.mean(next(iter(frames.values()))["n_eval"]))
        n_train = meta["n_rows"] - n_test
        corr_factor = 1.0 / 10 + n_test / n_train
        print(f"\n===== {short} (n_test均={n_test:.1f}, n_train均={n_train:.1f}, "
              f"NB 方差系数 1/k+n_te/n_tr={corr_factor:.4f} ≈ {corr_factor*10:.2f}x 朴素) =====")

        cids = [f"{short}_{m}_{s}_v1" for m in MODEL_KEYS for s in STRATEGIES]
        for metric in METRICS:
            mat = {cid: frames[cid][metric].to_numpy(dtype=float) for cid in cids}
            rows = []
            for a, b in combinations(cids, 2):     # 全配对 C(16,2)=120, 固定字典序防重复
                dbar, t, p, note = nb_paired_t(mat[a], mat[b], n_test, n_train)
                rows.append(dict(
                    dataset=meta["dataset_d"], metric=metric,
                    config_a=a, config_b=b,
                    mean_a=float(mat[a].mean()), mean_b=float(mat[b].mean()),
                    mean_diff=dbar, k=10,
                    n_test=n_test, n_train=n_train, var_corr_factor=round(corr_factor, 6),
                    t=t, df=9, p_raw=p, p_holm=np.nan, sig=sig_mark(p), note=note,
                ))
            holm_p = holm([r["p_raw"] for r in rows])
            for r, hp in zip(rows, holm_p):
                r["p_holm"] = float(hp) if np.isfinite(hp) else np.nan
            pairs = pd.DataFrame(rows)
            pairs.to_csv(out_dir / f"{short}_ttest_pairs_{metric}_v1.csv", index=False)

            # 16x16 下三角速览(原始 p + 星号; 对角线 --)
            lab = [c.replace(f"{short}_", "").replace("_v1", "") for c in cids]
            M = pd.DataFrame("--", index=lab, columns=lab, dtype=object)
            for r in rows:
                la = r["config_a"].replace(f"{short}_", "").replace("_v1", "")
                lb = r["config_b"].replace(f"{short}_", "").replace("_v1", "")
                cell = "--" if not np.isfinite(r["p_raw"]) else f"{r['p_raw']:.3g}{r['sig']}"
                if r["note"] == "identical":
                    cell = "1(ident)"
                M.loc[la, lb] = cell
            M.to_csv(out_dir / f"{short}_ttest_matrix_{metric}_v1.csv", encoding="utf-8-sig")

            if metric == "f1":                     # 关键对照只并一次(下面按 4 指标全量补)
                pass
            # 关键对照行(每指标都收): {model}_{strat} vs {model}_plain
            for m in MODEL_KEYS:
                for s in KEY_STRATS:
                    a, b = f"{short}_{m}_{s}_v1", f"{short}_{m}_plain_v1"
                    dbar, t, p, note = nb_paired_t(mat[a], mat[b], n_test, n_train)
                    key_rows.append(dict(
                        dataset=meta["dataset_d"], metric=metric,
                        comparison=f"{m}_{s}_vs_plain",
                        mean_a=float(mat[a].mean()), mean_b=float(mat[b].mean()),
                        mean_diff=dbar, t=t, df=9, p_raw=p,
                        sig=sig_mark(p), note=note,
                    ))

    key = pd.DataFrame(key_rows)
    key["comparison"] = key["dataset"].map({"credit-g": "creditg", "sick": "sick"}) + "_" + key["comparison"]
    key.to_csv(out_dir / "key_comparisons_v1.csv", index=False)

    # 终端速览: f1 的策略 vs plain(全部) —— 供运行记录与 D 抽读
    view = key[(key["metric"] == "f1")]
    for dsd in ("credit-g", "sick"):
        sub = view[view["dataset"] == dsd]
        print(f"\n--- f1 策略 vs plain 关键对照 ({dsd}) ---")
        for _, r in sub.iterrows():
            p = f"{r['p_raw']:.4g}" if np.isfinite(r["p_raw"]) else "NaN"
            print(f"  {r['comparison']:<28} Δf1={r['mean_diff']:+.4f}  t={r['t']:+.3f}  p={p}{r['sig']}"
                  + (f"  [{r['note']}]" if r["note"] else ""))
    print(f"\n[saved] results/ttest/: 每数据集 4 指标 pairs+matrix, 另 key_comparisons_v1.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="C5 NB-corrected fold-level paired t-test")
    ap.add_argument("--self-test-only", action="store_true", help="只跑核验锚点")
    args = ap.parse_args()
    if args.self_test_only:
        folds_all = {short: load_folds(short) for short in DATASETS}
        sys.exit(1 if self_test(folds_all) else 0)
    run()
