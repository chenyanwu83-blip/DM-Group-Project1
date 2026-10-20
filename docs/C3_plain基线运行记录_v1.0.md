# C3 plain 基线矩阵 · 运行记录 v1.0

- **负责人**: C（建模与平台）
- **日期**: 2026-10-14
- **状态**: 已完成自检，交 D 校验（→ D3 基线指标表 v1）
- **运行环境**: `/opt/anaconda3/envs/cds535/bin/python`（pandas 3.0.6 / scikit-learn 1.9.1 / imbalanced-learn 0.14.2，与 requirements.txt 锁定版本逐一相符）
- **入口**: `python -m src.run_experiment --baseline`（仓库根目录；单配置冒烟 `--dataset creditg --model lr --protocol cv10`）
- **协议**: 全部消费库内锁定切分（`cv_folds_*.csv` / `*_idx_*.csv`），指标一律 `src/evaluate.py` 的 `compute_metrics / aggregate_folds`；每折全新实例折内拟合，holdout 锁定配置 fit 一次评一次；`data_sha256` 取自 `split_manifest.json` 现值

## 1. 交付物清单

- 8 个 plain 配置（4 模型 × 2 数据集）× 2 协议（cv10 / holdout）：
  `results/{config_id}_run1_folds.csv` + `_run1_summary.csv`（cv10，D2 原文命名）
  `results/{config_id}_run1_holdout_folds.csv` + `_run1_holdout_summary.csv`（holdout；config_id 不含协议，加 `_holdout` 中缀防覆盖——D2 未显式规定 holdout 文件名，此为最小扩展，请 D 确认）
- 2 个 ZeroR 参照 × 2 协议：`{creditg,sick}_zero_ref_v1_*.csv`
- 预测留档：`results/preds/{config_id}_run1_{cv10,holdout}_preds.csv`（row_idx / y_true / y_pred / proba_pos），共 20 份，供 D3 方向核查
- config_id 全部按 D2 命名：`creditg_dt_plain_v1` … `sick_svc_plain_v1`；模板两文件保持空表头

## 2. 结果总表（accuracy 为 accuracy_mean；cv10 附 std）

### credit-g（正类 bad；cost 越低越好，ZeroR=1.5）

| 配置 | 协议 | acc | recall | f1 | auc | kappa | balanced_acc | cost |
|---|---|---|---|---|---|---|---|---|
| dt_plain | cv10 | 0.7140±0.0258 | 0.2700 | 0.3218 | 0.6996 | 0.1911 | 0.5871 | 1.1620 |
| dt_plain | holdout | 0.7167 | 0.3333 | 0.4138 | 0.6952 | 0.2411 | 0.6071 | 1.0833 |
| rf_plain | cv10 | **0.7630**±0.0283 | 0.4000 | 0.4988 | 0.7931 | 0.3593 | 0.6593 | 0.9570 |
| rf_plain | holdout | 0.7333 | 0.3333 | 0.4286 | 0.7755 | 0.2727 | 0.6190 | 1.0667 |
| lr_plain | cv10 | 0.7520±0.0275 | **0.4700** | **0.5307** | 0.7846 | **0.3667** | **0.6714** | **0.8840** |
| lr_plain | holdout | 0.7400 | 0.4778 | 0.5244 | 0.7828 | 0.3478 | 0.6651 | **0.8867** |
| svc_plain | cv10 | **0.7680**±0.0331 | 0.3933 | 0.4979 | **0.7949** | 0.3649 | 0.6610 | 0.9600 |
| svc_plain | holdout | 0.7600 | 0.3889 | 0.4930 | 0.7781 | 0.3502 | 0.6540 | 0.9733 |
| zero_ref | cv10 | 0.7000±0.0000 | 0 | 0 | 0.5000 | 0 | 0.5000 | 1.5000 |
| zero_ref | holdout | 0.7000 | 0 | 0 | 0.5000 | 0 | 0.5000 | 1.5000 |

### sick（正类 sick；无官方代价矩阵，cost 列空）

| 配置 | 协议 | acc | recall | f1 | auc | kappa | balanced_acc |
|---|---|---|---|---|---|---|---|
| dt_plain | cv10 | 0.9822±0.0050 | 0.7491 | 0.8362 | 0.9811 | 0.8271 | 0.8733 |
| dt_plain | holdout | 0.9867 | 0.8116 | 0.8819 | 0.9934 | 0.8749 | 0.9049 |
| rf_plain | cv10 | 0.9854±0.0049 | 0.8100 | **0.8703** | 0.9975 | 0.8627 | 0.9034 |
| rf_plain | holdout | 0.9859 | **0.8551** | 0.8806 | **0.9982** | 0.8731 | 0.9247 |
| lr_plain | cv10 | 0.9666±0.0090 | 0.6063 | 0.6867 | 0.9499 | 0.6696 | 0.7982 |
| lr_plain | holdout | 0.9655 | 0.6232 | 0.6880 | 0.9655 | 0.6700 | 0.8055 |
| svc_plain | cv10 | 0.9740±0.0069 | 0.6842 | 0.7625 | 0.9788 | 0.7491 | 0.8386 |
| svc_plain | holdout | 0.9700 | 0.6522 | 0.7258 | 0.9849 | 0.7101 | 0.8214 |
| zero_ref | cv10 | 0.9388±0.0008 | 0 | 0 | 0.5000 | 0 | 0.5000 |
| zero_ref | holdout | 0.9390 | 0 | 0 | 0.5000 | 0 | 0.5000 |

初步观察（供 D3 / 报告 ch5，非结论）：
- credit-g 全部模型 cost 均低于 ZeroR 的 1.5；**LR 的 cost 最优（0.884）**——线性模型 + OneHot 对少数类 recall 最高（0.47）所致；这正是 RQ2 策略实验的参照原点；
- sick 上树模型（RF/DT）全面领先（f1 0.87/0.84），LR 最弱（recall 0.61）；
- 两条协议方向一致，无符号翻转。

## 3. 锚点核对（全过才认为评估链路可信）

| 锚点 | 预期 | 实测 | 结果 |
|---|---|---|---|
| ZeroR cv10 credit-g acc | = 0.7000（700/1000） | 0.7000 | ✅ |
| ZeroR cv10 credit-g cost | = 1.5（5×300/1000） | 1.5000 | ✅ |
| ZeroR cv10 sick acc | ≈ 0.9388（3541/3772） | 0.9388 | ✅ |
| ZeroR holdout credit-g acc | = 0.7000（210/300） | 0.7000 | ✅ |
| ZeroR holdout sick acc | ≈ 0.9390（1063/1132） | 0.9390 | ✅ |
| ZeroR f1/auc/kappa | 0 / 0.5 / 0 | 0 / 0.5 / 0 | ✅ |
| DT vs RUN_GUIDE 锚点（PreExperiment 默认 gini 树 0.6660±0.0233） | 同量级（规格树参数不同，数字允许略异） | 0.7140±0.0258 | ✅ |
| DT vs RUN_GUIDE 锚点（sick 0.9867±0.0049） | 同量级 | 0.9822±0.0050 | ✅ |

## 4. 落盘自查（7 项，脚本化核验，PASS）

1. 20 组文件成对齐全（10 配置 × 2 协议的 folds + summary），命名与 D2 一致；
2. folds 列 = fold_template 28 列、summary 列 = template 46 列，零偏差（reindex 保证）；
3. cv10 行数=10 且 n_folds=10；holdout 行数=1 且 n_folds=1；
4. holdout 全部 `*_std` 列为空（NaN），无 0 冒充标准差；
5. 全部 `data_sha256` = split_manifest.json 现值（credit-g d0baf9fd…/ sick 3cc14348…）；
6. credit-g 行均有 cost_mean/cost_pooled；sick 行两列均空；
7. 模板与 6 个切分索引未被扰动：`python src/evaluate.py --check-data --prepare-holdout` 全部报 EXISTS identical / unchanged。

preds 留档 20 份行数全对（cv10=1000/3772，holdout=300/1132）。

## 5. 概率方向自检（D3 前置材料）

- credit-g × LR：`classes_ = ['bad' 'good']`；前 6 行中 bad 行概率 0.623 / 0.749，good 行 0.029–0.293 —— 正类方向正确；
- sick × DT：负类行概率 ≈ 0.0；正类（sick）行概率 0.989（6 行中 5 行，1 行 0.122 为模型误判而非方向错误）—— 方向正确。

## 6. 已知事项

1. `SVC(probability=True)` 在 sklearn 1.9 触发 FutureWarning（1.11 移除）——已在 C2 回执 5.3 登记，锁定版本不受影响，运行日志中有原样记录；
2. holdout 输出文件名加 `_holdout` 中缀是对 D2 命名的最小扩展（config_id 不含协议），请 D 确认或给出正式命名；
3. 全部结果由同一解释器一次性产出（--baseline），SVC×sick 为最慢环节；如需复跑，任何机器同版本环境应得到逐位一致结果（seed=42 + 锁定切分 + 固定超参）。

## 7. D1 确认清单 —— C 部分（随 C2 一并声明）

- [x] cv10 std ddof=0（aggregate_folds 实现）
- [x] 索引/文件命名遵循 D2（holdout 中缀见 6.2 待确认）
- [x] 两协议样本范围各自独立，未混算
- [x] 折内拟合（每折全新实例，验证折只 transform）
- [x] 使用 D2 列名与接口；逐折数据与预测留档已提供（results/preds/）
