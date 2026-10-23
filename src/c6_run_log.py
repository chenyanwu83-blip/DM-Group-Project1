# -*- coding: utf-8 -*-
"""C6: 运行日志总表 —— 全部实验运行的单一索引（V3 C6 + V4 增补含集成 16 配置）。

用法(仓库根目录):
    python -m src.c6_run_log          # 扫描 results/ 全部 *_summary.csv 生成总表

口径:
    - 一行 = 一个 config × 一个协议 的实际运行(数据来自各 summary 文件本身,
      不重算任何指标): config_id / dataset / model / strategy / protocol /
      run_id / seed / data_sha256 / 关键参数 / runtime_sec / 三类文件路径;
    - 覆盖: 主矩阵 32 配置(plain/smote/cost/threshold × dt/rf/lr/svc × 2 数据集)
      + ZeroR 参照 2 配置 + 集成 8 配置(bag/ada × plain/smote × 2 数据集),
      各 × cv10/holdout 两协议;
    - 关键参数从 c_config / V4 §1.3 冻结规格代出, 与代码单一事实来源一致。
输出: results/run_log_v1.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.c_config import MODELS, DATASETS, RUN_ID, SEED

# 主矩阵 4 模型冻结超参(c_config 单一来源) → 一句话参数串
MAIN_PARAMS = {
    k: ",".join(f"{kk}={vv}" for kk, vv in v["kw"].items()) for k, v in MODELS.items()
}
MAIN_DISPLAY = {k.lower(): v["display"] for k, v in MODELS.items()}
ENS_PARAMS = {
    "bag": "Bagging(base=DT entropy,ccp_alpha=0.01; n_estimators=100, aligned RF)",
    "ada": "AdaBoost(base=DT entropy,ccp_alpha=0.01; n_estimators=100, aligned RF)",
}
ZERO_PARAMS = "DummyClassifier(strategy=most_frequent)"


def probe(folds, summary, preds):
    return all(p.exists() for p in (folds, summary, preds))


def row_for(protocol, summary_path, params):
    su = pd.read_csv(summary_path).iloc[0]
    sfx = "" if protocol == "cv10" else "_holdout"
    folds = ROOT / "results" / f"{su.config_id}_{su.run_id}{sfx}_folds.csv"
    summ = ROOT / "results" / f"{su.config_id}_{su.run_id}{sfx}_summary.csv"
    preds = ROOT / "results" / "preds" / f"{su.config_id}_{su.run_id}_{protocol}_preds.csv"
    if "ensemble" in str(summary_path):
        folds = ROOT / "results" / "ensemble" / f"{su.config_id}_{su.run_id}{sfx}_folds.csv"
        summ = ROOT / "results" / "ensemble" / f"{su.config_id}_{su.run_id}{sfx}_summary.csv"
        preds = ROOT / "results" / "ensemble" / "preds" / f"{su.config_id}_{su.run_id}_{protocol}_preds.csv"
    rel = lambda p: str(p.relative_to(ROOT))
    return dict(
        config_id=su.config_id, dataset=su.dataset, model=su.model, strategy=su.strategy,
        protocol=protocol, run_id=su.run_id, seed=su.seed, data_sha256=su.data_sha256,
        params=params + f"; strategy={su.strategy}; seed={SEED}",
        runtime_sec=round(float(su.runtime_sec), 3),
        folds_csv=rel(folds), summary_csv=rel(summ), preds_csv=rel(preds),
        files_complete=probe(folds, summ, preds),
    )


def run():
    rows = []
    # ① 主矩阵 32 配置 + ② ZeroR 参照(主矩阵命名空间)
    for p in sorted((ROOT / "results").glob("*_v1_run1*_summary.csv")):
        stem = p.name
        protocol = "holdout" if "_holdout_" in stem else "cv10"
        cid = stem.split("_run1")[0]                     # {ds}_{model}_{strategy}_v1 或 {ds}_zero_ref_v1
        assert cid.endswith("_v1"), f"意外命名: {stem}"
        core = cid[:-3]                                   # 去掉 _v1 后缀
        if core.endswith("_zero_ref"):
            ds, mk = core[: -len("_zero_ref")], "zero"
            params = ZERO_PARAMS
        else:
            ds, rest = core.split("_", 1)                # ds = 第一段
            mk = rest.split("_", 1)[0]                   # dt/rf/lr/svc
            assert mk in MAIN_DISPLAY, f"主矩阵命名空间混入未知模型: {stem}"
            params = f"{MAIN_DISPLAY[mk]}({MAIN_PARAMS[mk.upper()]})"
        rows.append(row_for(protocol, p, params))
    # ③ 集成 8 配置(独立命名空间, V4 §1.3)
    for p in sorted((ROOT / "results" / "ensemble").glob("*_v1_run1*_summary.csv")):
        stem = p.name
        protocol = "holdout" if "_holdout_" in stem else "cv10"
        cid = stem.split("_run1")[0]
        ds, mk = cid.split("_")[0], cid.split("_")[1]
        rows.append(row_for(protocol, p, ENS_PARAMS[mk]))

    out = pd.DataFrame(rows)
    out_path = ROOT / "results" / "run_log_v1.csv"
    out.to_csv(out_path, index=False)
    n_main = len(out[out.config_id.str.contains(r"_(?:dt|rf|lr|svc)_")])
    n_zero = len(out[out.config_id.str.contains("zero")])
    n_ens = len(out[out.config_id.str.contains(r"_(?:bag|ada)_")])
    print(f"[C6] 总表 {len(out)} 行 = 主矩阵 {n_main}(预期64) + ZeroR {n_zero}(预期4) "
          f"+ 集成 {n_ens}(预期16); files_complete 全 True: {bool(out.files_complete.all())}")
    assert n_main == 64 and n_zero == 4 and n_ens == 16, "行数不符, 拒绝交付"
    assert out.files_complete.all(), "存在缺文件的运行, 拒绝交付"
    assert (out.seed == SEED).all()
    print(f"[saved] results/run_log_v1.csv | 列: {list(out.columns)}")


if __name__ == "__main__":
    run()
