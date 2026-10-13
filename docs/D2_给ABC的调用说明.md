# D2：给A/B/C的调用说明

在仓库根目录运行 `python src/evaluate.py` 自检。该命令生成或检查空模板，不训练模型、不生成正式成绩。D的协议是 `docs/D1_评估协议_v1.0.md`；D1中提议的标准差、索引路径及协议样本范围，先全组确认。

## 计算一折或一次留出指标

下面只是接入示例；`pipe`、`X_valid`、`y_valid` 是C/A/B的实验代码中已有变量，不是一个可以独立运行的模型训练脚本。

```python
from src.evaluate import compute_metrics

# 先用C/A/B的实验代码fit训练折，然后对当前验证折预测。
y_pred = pipe.predict(X_valid)
proba = pipe.predict_proba(X_valid)
metrics = compute_metrics(
    y_valid, y_pred, proba,
    pos_label="bad", dataset="credit-g", classes=pipe.classes_,
)
print(metrics)
```

sick使用 `pos_label="sick", dataset="sick"`，且y来自大写 `Class`；该CSV里小写 `sick` 是特征列：[sick原数据](https://raw.githubusercontent.com/chenyanwu83-blip/DM-Group-Project1/main/data/raw/sick.csv)。

二维概率输入必须同时给 `classes`，脚本定位正类对应列。若给一维数组，它必须已经是正类概率，函数不能从一维数组推断你取对了哪列。AUC真实标签已显式映射成正类=1，符合 [scikit-learn AUC输入约定](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html)。

## 保存逐折和汇总记录

推荐共用D生成的CV行位置，避免不同人划分不同。实际训练还是由C的预处理/采样/模型Pipeline负责；本工具不替换它。

```python
from time import perf_counter
from sklearn.base import clone
import pandas as pd
from src.evaluate import compute_metrics, aggregate_folds, FOLD_COLUMNS

# 接入示例：X/y必须来自完整原始CSV的同一行序；pipe由C构建。
dataset = "credit-g"
config_id = "creditg_dt_plain_v1"
model_name = "DecisionTree"
strategy = "plain"
run_id = "run1"
data_hash = "请读取results/split_manifest.json中的当前数据哈希"
folds = pd.read_csv(f"results/cv_folds_{dataset}.csv")
assert len(folds) == len(X) == len(y)
assert folds["row_idx"].tolist() == list(range(len(X)))
records, raw_rows = [], []
start = perf_counter()
for fold_id in range(1, 11):
    test_idx = folds.loc[folds["fold_id"] == fold_id, "row_idx"].to_numpy()
    train_idx = folds.loc[folds["fold_id"] != fold_id, "row_idx"].to_numpy()
    estimator = clone(pipe)
    fold_start = perf_counter()
    estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
    pred = estimator.predict(X.iloc[test_idx])
    prob = estimator.predict_proba(X.iloc[test_idx])
    m = compute_metrics(
        y.iloc[test_idx], pred, prob,
        dataset=dataset, classes=estimator.classes_,
    )
    records.append(m)
    raw = {
        **m, "config_id": config_id, "run_id": run_id,
        "dataset": dataset, "model": model_name, "strategy": strategy,
        "protocol": "cv10", "fold_id": fold_id, "seed": 42,
        "data_sha256": data_hash,
        "runtime_sec": perf_counter() - fold_start,
    }
    raw_rows.append({key: raw.get(key) for key in FOLD_COLUMNS})

summary = aggregate_folds(
    records, config_id=config_id, dataset=dataset, model=model_name,
    strategy=strategy, protocol="cv10", run_id=run_id,
    data_sha256=data_hash, runtime_sec=perf_counter() - start,
)
pd.DataFrame(raw_rows, columns=FOLD_COLUMNS).to_csv(
    f"results/{config_id}_{run_id}_folds.csv", index=False)
pd.DataFrame([summary]).reindex(
    columns=pd.read_csv("results/template.csv", nrows=0).columns
).to_csv(f"results/{config_id}_{run_id}_summary.csv", index=False)
```

每个配置用自己的 `config_id` 和文件名，不覆盖别人的输出。所有10折指标先进入 `records`，再做一次汇总；不要把不同配置、不同数据集、不同run或不同协议塞进同一个 `records`。阈值策略需要训练折内选择阈值后锁定，因此示例中的默认 `predict()` 仅适用于已实现正确决策规则的Pipeline。

## 留出评价

读取自己的数据集索引：credit-g使用 `train_idx_creditg.csv` / `holdout_idx_creditg.csv`，sick使用 `train_idx_sick.csv` / `holdout_idx_sick.csv`。以 `.iloc[row_idx]` 选择原始CSV相同行位置；训练集fit，留出集predict一次。

```python
summary = aggregate_folds(
    [holdout_metrics], config_id=config_id, dataset=dataset,
    model=model_name, strategy=strategy, protocol="holdout",
    run_id=run_id, data_sha256=data_hash, runtime_sec=elapsed_seconds,
)
```

`holdout_metrics`来自一次 `compute_metrics()`；`elapsed_seconds`是实际测得的耗时。这些变量需由C的实验脚本赋值；留出std列自动为空，不造标准差。

## 模板列说明

- `config_id/run_id/dataset/model/strategy/protocol/seed/data_sha256`：来源定位。
- 八项指标及cost：分别保存 `*_mean`、`*_std`、`*_n_valid`。代价仅credit-g有效。
- `TN/FP/FN/TP` 和 `confusion_matrix`：CV汇总折计数；holdout是单次计数；矩阵始终为 `[[TN,FP],[FN,TP]]`。
- `cost_pooled`：总FN/FP按总N计算；不要和折平均的 `cost_mean` 混用。
- `precision_undefined_folds/recall_undefined_folds`：分母零情况计数。
- `runtime_sec/notes`：实际耗时与不能计算的原因，不能把“空值”替换成0。

保留 `fold_template.csv` 格式的逐折数据供D核查和C统计比较；概率、预测标签与测试行位置也建议另存，便于D3验证方向。模板本身始终是空表头，真实结果写进新文件。
