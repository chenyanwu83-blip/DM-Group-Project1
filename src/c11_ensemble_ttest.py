# -*- coding: utf-8 -*-
"""C11: 集成研究 2 对预指定折级配对 t 检验（V4 方案 §1.4，exploratory 定位）。

用法(仓库根目录):
    python -m src.c11_ensemble_ttest        # 2 对 × 两侧 = 4 个检验(f1, NB 校正)

预指定的配对(每数据集, 同折配对 10 折, 其余集成结果一律只报均值±标准差):
    ① {ds}_bag_plain  vs 主矩阵 {ds}_dt_plain   —— 方差缩减(bagging)单独贡献多少
    ② {ds}_ada_plain  vs 主矩阵 {ds}_dt_smote   —— 重加权(boosting) vs 重采样(SMOTE)

纪律(V4 §1.2 决策②):
    - 只做这 2 对, 不做集成的其他任何配对, 防多重比较质疑;
    - 报告标准措辞: "reported as an exploratory extension; only two
      pre-specified comparisons were tested";
    - 检验方法与 C5 完全同款(NB 校正 + 边界不造数), 复用 src/stats_test.py 的
      nb_paired_t / sig_mark —— 公式单一事实来源, 不另写一份。

输出(入 results/ttest/, 与 C5 同命名空间):
    results/ttest/ensemble_2pairs_prespecified_v1.csv
      列(17): dataset, comparison, metric, label, mean_a, mean_b, mean_diff,
              k, t, df, p_raw, sig, note, n_test, n_train, var_corr_factor,
              method
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.c_config import DATASETS
from src.stats_test import nb_paired_t, sig_mark, METRICS  # 复用, 不复制公式

PAIRS = [  # (集成侧 config, 主矩阵侧 config, 语义标签)
    ("bag_plain", "dt_plain", "bagging方差缩减 vs 单树基线"),
    ("ada_plain", "dt_smote", "boosting重加权 vs SMOTE重采样"),
]
PRIMARY_METRIC = "f1"   # 预指定主指标; 其余指标按 V4 §1.2 只作描述性均值±标准差


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
        # 锁定折表每折行数(权威), 与 folds 文件 n_eval 对齐断言在配对前完成
        neval_authority = pd.read_csv(
            ROOT / f"results/{short}_dt_plain_v1_run1_folds.csv")["n_eval"].tolist()
        for ens_key, main_key, label in PAIRS:
            ens = load_cv10_folds(ROOT / "results" / "ensemble" / f"{short}_{ens_key}_v1_run1_folds.csv")
            main = load_cv10_folds(ROOT / f"results/{short}_{main_key}_v1_run1_folds.csv")
            # 同折配对前置断言: 两表折大小逐位一致(=同一锁定切分), seed/sha 同源
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
    # 其余指标仅描述性(均值±std 已在 ensemble summary 与 C5 关系之外, 不做检验)
    for short, meta in DATASETS.items():
        for m in ("bag", "ada"):
            for s in ("plain", "smote"):
                su = pd.read_csv(ROOT / "results" / "ensemble" / f"{short}_{m}_{s}_v1_run1_summary.csv").iloc[0]
                print(f"  [描述性] {short}_{m}_{s}: f1={su.f1_mean:.4f}±{su.f1_std:.4f} "
                      f"auc={su.auc_mean:.4f} (不做检验)")


if __name__ == "__main__":
    run()
