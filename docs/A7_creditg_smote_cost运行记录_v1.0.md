# A7 运行记录 v1.0 — credit-g 侧 smote + cost 策略矩阵（2026-10-20）

> 角色 A 交付（V3 方案 A7：credit-g 侧实验矩阵）。plain（C3，10-14）与 threshold（C4，10-15）已在库；
> 本次补齐 smote 与 cost 两组，credit-g 侧 4 策略 × 4 模型 = 16 配置就此齐全，按契约（10-21）交 D 做 D4 全量汇总。

## 执行方式

- 入口：C 的统一入口单配置模式（不用 `--matrix`，避免连跑 sick 侧与 B 撞车）：
  `python -m src.run_experiment --dataset creditg --model {dt,lr,rf,svc} --strategy {smote,cost}`
- 每配置自动双协议：cv10（锁定折表）+ holdout（锁定 70:30 索引），指标走 evaluate.py，输出列=模板，preds 留档。
- cost 组由 c_pipelines 注入冻结规格的官方 5:1（class_weight={'bad':5,'good':1}）；smote 组 SMOTE(random_state=42) 在 imblearn Pipeline 内仅训练折执行。
- 环境：cds535 venv（requirements.txt 锁定版），seed=42，config_id 延续 D2 命名 `creditg_{model}_{strategy}_v1_run1`。

## 落盘自查（16 配置 × 2 协议）

| # | 检查项 | 结果 |
|---|---|---|
| 1 | 新增 8 配置 × {folds, summary, holdout_folds, holdout_summary} 共 32 份文件齐全 | PASS |
| 2 | summary 列与 template.csv 零偏差（46 列） | PASS |
| 3 | cv10 折数=10；holdout 单行且 std 列为 NaN（不冒充 0） | PASS |
| 4 | data_sha256 与 split_manifest.json 一致 | PASS |
| 5 | preds 留档 16 份（cv10+holdout，含 row_idx/y_true/y_pred/proba_pos） | PASS |
| 6 | 不覆盖既有文件（plain/threshold/A6 输出零改动） | PASS |
| 7 | 策略方向合理性：smote/cost 的 recall(bad) 相对 plain 全部上升、总代价全部下降 | PASS |

## 关键结果（cv10 均值，正类 bad；完整数字见 results/*_summary.csv）

| 配置 | acc | recall | F1 | AUC | cost |
|---|---|---|---|---|---|
| lr_plain（参照） | 0.752 | 0.470 | 0.531 | 0.785 | 0.884 |
| lr_smote | 0.716 | 0.683 | **0.591** | 0.780 | 0.664 |
| rf_smote | 0.738 | 0.457 | 0.509 | 0.785 | 0.914 |
| svc_smote | 0.731 | 0.570 | 0.560 | 0.784 | 0.785 |
| dt_smote | 0.670 | 0.500 | 0.471 | 0.690 | 0.930 |
| lr_cost | 0.650 | 0.863 | 0.597 | 0.782 | **0.514** |
| rf_cost | 0.696 | 0.743 | 0.595 | 0.797 | 0.612 |
| svc_cost | 0.686 | 0.760 | 0.593 | 0.779 | 0.602 |
| dt_cost | 0.525 | 0.863 | 0.523 | 0.704 | 0.639 |

holdout 方向与 cv10 一致（recall↑ cost↓），如 lr_cost holdout：acc 0.643 / recall 0.822 / cost_pooled 0.570。

## 初步观察（供 D4/D6 与报告 ch6 引用，最终以 D 汇总与 C5 t 检验为准）

1. **RQ3 素材**：官方 5:1 代价矩阵下全模型 recall 大涨（+0.34~+0.59）、总代价普降；按总代价与 F1 双口径，lr_cost 均居首（cost 0.514 / F1 0.597）。plain 最优（lr, F1 0.531）与 cost 设定最优（lr_cost）模型一致，但第 2-4 名排序洗牌——「最优模型是否改变」需 sick 侧与 t 检验补齐后回答。
2. **RQ2 素材**：dt_cost 是"accuracy 崩塌但代价大幅改善"的极端案例（acc 0.525 vs plain 0.714，cost 1.162→0.639），70% 多数类基线在代价视角下完全失效。
3. **RQ1 素材**：smote 的增益模型间差异大（lr F1 +0.060 最大、rf 仅 +0.010），线性模型从重采样获益最多；AUC 各策略基本稳定（±0.01 量级）。

## 待办移交

- D：D4 全量汇总（credit-g 16 配置已齐；sick 侧 smote/cost 仍缺，等 B 的 B7）。
- C：C5 的 32 配置 t 检验需等 sick 侧补齐后运行。
- A6 复核（D 审口径）进行中。
