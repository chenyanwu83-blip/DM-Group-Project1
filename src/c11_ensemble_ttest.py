"""
集成研究里事先说好的 2 对配对 t 检验(f1 指标, NB 校正):
    ① bag_plain vs 主矩阵 dt_plain —— bagging 到底比单棵树好多少
    ② ada_plain vs 主矩阵 dt_smote —— boosting 的重加权 vs SMOTE 重采样

只做这 2 对, 集成的其他对比一律只报均值±标准差(不然多重比较说不过去)。
检验直接复用 stats_test.py 里的 nb_paired_t 和 sig_mark, 不另抄一份公式。

    python -m src.c11_ensemble_ttest
结果: results/ttest/ensemble_2pairs_prespecified_v1.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.c_config import DATASETS
from src.stats_test import nb_paired_t, sig_mark, METRICS   # 直接用 C5 的函数, 不复制

PAIRS = [   # (集成侧, 主矩阵侧, 这对比较想回答什么)
    ("bag_plain", "dt_plain", "bagging方差缩减 vs 单树基线"),
    ("ada_plain", "dt_smote", "boosting重加权 vs SMOTE重采样"),
]
PRIMARY_METRIC = "f1"   # 主指标定的是 f1, 其他指标只描述不检验


def load_cv10_folds(path):
    f = pd.read_csv(path)
    assert sorted(f["fold_id"]) == list(range(1, 11)), f"{path}: fold_id 非 1..10"
    assert (f["protocol"] == "cv10").all(), f"{path}: 混入非 cv10"
    return f


def run():
    rows = []
    for short, meta in DATASETS.items():
        n_test = float(np.mean(pd.read_csv(
            ROOT / "results" / f"cv_folds_{meta['dataset_d']}.csv").groupby("fold_id").size()))
        # 折表算每折行数当基准, 下面配对前先对一遍
        neval_authority = pd.read_csv(
            ROOT / f"results/{short}_dt_plain_v1_run1_folds.csv")["n_eval"].tolist()
        for ens_key, main_key, label in PAIRS:
            ens = load_cv10_folds(ROOT / "results" / "ensemble" / f"{short}_{ens_key}_v1_run1_folds.csv")
            main = load_cv10_folds(ROOT / f"results/{short}_{main_key}_v1_run1_folds.csv")
            # 配对前提: 两边折大小完全一致(同一套锁定切分), seed/sha 也一样
            assert ens["n_eval"].tolist() == main["n_eval"].tolist() == neval_authority, \
                f"{short} {ens_key} vs {main_key}: 折大小不一致, 无同折配对资格"
            assert (ens["seed"] == 42).all() and (main["seed"] == 42).all()
            assert (ens["data_sha256"] == main["data_sha256"]).all()
            n_train = meta["n_rows"] - n_test
            dbar, t, p, note = nb_paired_t(
                ens[PRIMARY_METRIC].to_numpy(float),
                main[PRIMARY_METRIC].to_numpy(float), n_test, n_train)
            rows.append(dict(
                dataset=meta["dataset_d"],
                comparison=f"{short}_{ens_key}_vs_{main_key}",
                metric=PRIMARY_METRIC, label=label,
                mean_a=float(ens[PRIMARY_METRIC].mean()),
                mean_b=float(main[PRIMARY_METRIC].mean()),
                mean_diff=dbar, k=10, t=t, df=9, p_raw=p, sig=sig_mark(p),
                note=note, n_test=n_test, n_train=n_train,
                var_corr_factor=round(1 / 10 + n_test / n_train, 6),
                method="NB-corrected fold-level paired t (same as C5)",
            ))
    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "ttest" / "ensemble_2pairs_prespecified_v1.csv"
    out.to_csv(out_path, index=False)

    print(f"[C11] 预指定 2 对 × 两侧 = {len(out)} 个检验 (指标=f1, NB 校正同 C5):")
    for _, r in out.iterrows():
        p = f"{r.p_raw:.4g}" if np.isfinite(r.p_raw) else "NaN"
        print(f"  {r.comparison:<38} Δf1={r.mean_diff:+.4f}  t={r.t:+.3f}  p={p}{r.sig}"
              + (f"  [{r.note}]" if r.note else ""))
    print('  标准措辞: "reported as an exploratory extension; only two '
          'pre-specified comparisons were tested"')
    print(f"[saved] {out_path.relative_to(ROOT)}")
    # 其余指标只打出来看一眼, 不做检验
    for short, meta in DATASETS.items():
        for m in ("bag", "ada"):
            for s in ("plain", "smote"):
                su = pd.read_csv(ROOT / "results" / "ensemble" / f"{short}_{m}_{s}_v1_run1_summary.csv").iloc[0]
                print(f"  [描述性] {short}_{m}_{s}: f1={su.f1_mean:.4f}±{su.f1_std:.4f} "
                      f"auc={su.auc_mean:.4f} (不做检验)")


if __name__ == "__main__":
    run()
