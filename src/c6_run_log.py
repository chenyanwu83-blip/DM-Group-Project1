"""
把 results/ 里所有 *_summary.csv 扫一遍, 汇总成一张运行日志总表
results/run_log_v1.csv。一行就是一次实际跑过的实验(config × 协议),
所有信息直接从 summary 文件里读, 一个指标都不重算。

覆盖主矩阵 32 配置 + ZeroR 2 个 + 集成 8 个, 每个都有 cv10/holdout 两条。

    python -m src.c6_run_log
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.c_config import MODELS, DATASETS, RUN_ID, SEED

# 参数串从 c_config 里拼出来, 免得手写第二遍写岔了
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
    # 主矩阵 32 配置和 ZeroR 参照都在 results/ 顶层
    for p in sorted((ROOT / "results").glob("*_v1_run1*_summary.csv")):
        stem = p.name
        protocol = "holdout" if "_holdout_" in stem else "cv10"
        cid = stem.split("_run1")[0]   # 形如 {ds}_{model}_{strategy}_v1 或 {ds}_zero_ref_v1
        assert cid.endswith("_v1"), f"意外命名: {stem}"
        core = cid[:-3]   # 去掉结尾的 _v1
        if core.endswith("_zero_ref"):
            ds, mk = core[: -len("_zero_ref")], "zero"
            params = ZERO_PARAMS
        else:
            ds, rest = core.split("_", 1)
            mk = rest.split("_", 1)[0]                   # dt/rf/lr/svc
            assert mk in MAIN_DISPLAY, f"results/ 顶层混进了不认识的模型: {stem}"
            params = f"{MAIN_DISPLAY[mk]}({MAIN_PARAMS[mk.upper()]})"
        rows.append(row_for(protocol, p, params))
    # 集成的 8 个配置单独放在 results/ensemble/
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
