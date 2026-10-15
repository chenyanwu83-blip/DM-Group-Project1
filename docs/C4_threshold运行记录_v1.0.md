# C4 threshold 策略 · 运行记录 v1.0

- **负责人**: C（建模与平台）
- **日期**: 2026-10-15
- **状态**: 本机完成 + 自检 PASS，**尚未 commit/push**（待组内过目后再定提交切分）
- **运行环境**: `/opt/anaconda3/envs/cds535/bin/python`（pandas 3.0.6 / scikit-learn 1.9.1 / imbalanced-learn 0.14.2，锁定版本）
- **入口**: `python -m src.run_experiment --matrix --strategy threshold`（仓库根；单配置冒烟 `--dataset creditg --model lr --strategy threshold --protocol cv10` 先行通过）

## 1. 交付物清单

| 文件 | 变更 | 内容 |
|---|---|---|
| `src/c4_threshold.py` | 新增 | `ThresholdClassifier` 包装器：内层 StratifiedKFold(5, shuffle, rs=42) 产 OOF 正类概率 → τ 网格 0.05–0.95 步 0.05（19 档）按 max-F1 选 τ → 锁定 → 全训练折重拟合；`predict()` 按 τ 判（D2：predict 体现锁定决策规则）、`predict_proba()` 透传 |
| `src/c_pipelines.py` | 修改 | 原 `build_pipe` 主体抽为 `_assemble`（plain/smote/cost）；`build_pipe` 的 threshold 分支返回包装器，基座为 **plain 子流水线**（每折全新工厂注入） |
| `src/run_experiment.py` | 修改 | ① `--strategy` choices 增 `threshold`；② 新增 `--matrix`（按给定策略跑 8 配置×2 协议）；③ 锁定 τ 与内层 OOF F1 记入 folds 的 `notes` 列 |
| `results/` | 新增 32 份 | 8 配置 × 2 协议的 folds + summary（命名 `{ds}_{model}_threshold_v1[_holdout]_*`，与 C3 同款） |
| `results/preds/` | 新增 16 份 | `*_threshold_v1_run1_{cv10,holdout}_preds.csv`（row_idx/y_true/y_pred/proba_pos） |

## 2. 结果总表（cv10 附每折锁定 τ 范围）

### credit-g（正类 bad；cost 越低越好，ZeroR=1.5，plain 最优 cost=0.884@LR）

| 配置 | 协议 | acc | recall | f1 | auc | cost | τ |
|---|---|---|---|---|---|---|---|
| dt_threshold | cv10 | 0.6290 | 0.7933 | 0.5610 | 0.6996 | 0.6190 | 0.20–0.40 |
| dt_threshold | holdout | 0.6000 | 0.8000 | 0.5455 | 0.6952 | 0.6400 | 0.25 |
| rf_threshold | cv10 | 0.6880 | 0.7567 | **0.5946** | 0.7931 | **0.6040** | 0.25–0.40 |
| rf_threshold | holdout | 0.6300 | 0.8556 | 0.5811 | 0.7755 | **0.5433** | 0.25 |
| lr_threshold | cv10 | 0.7010 | 0.7200 | 0.5908 | 0.7846 | 0.6350 | 0.20–0.35 |
| lr_threshold | holdout | 0.7367 | 0.6556 | 0.5990 | 0.7828 | 0.6767 | 0.35 |
| svc_threshold | cv10 | 0.7250 | 0.6567 | 0.5903 | 0.7949 | 0.6870 | 0.25–0.35 |
| svc_threshold | holdout | 0.6933 | 0.6667 | 0.5660 | 0.7781 | 0.7067 | 0.30 |

### sick（正类 sick；无官方代价矩阵，cost 空；plain 最优 f1=0.8703@RF）

| 配置 | 协议 | acc | recall | f1 | auc | τ |
|---|---|---|---|---|---|---|
| dt_threshold | cv10 | 0.9788 | 0.7926 | 0.8203 | 0.9811 | 0.25–0.60 |
| dt_threshold | holdout | 0.9823 | 0.8696 | 0.8571 | 0.9934 | 0.25 |
| rf_threshold | cv10 | 0.9849 | 0.8792 | **0.8766** | 0.9975 | 0.35–0.50 |
| rf_threshold | holdout | 0.9885 | 0.8551 | **0.9008** | 0.9982 | 0.55 |
| lr_threshold | cv10 | 0.9677 | 0.7707 | 0.7453 | 0.9499 | 0.30–0.35 |
| lr_threshold | holdout | 0.9673 | 0.7246 | 0.7299 | 0.9655 | 0.35 |
| svc_threshold | cv10 | 0.9769 | 0.7926 | 0.8087 | 0.9788 | 0.30–0.40 |
| svc_threshold | holdout | 0.9682 | 0.6957 | 0.7273 | 0.9849 | 0.45 |

## 3. 与 plain 对照（cv10，关键列）

| 数据集 | 模型 | f1: plain→threshold | cost: plain→threshold | 说明 |
|---|---|---|---|---|
| credit-g | DT | 0.3218 → 0.5610 | 1.1620 → 0.6190 | recall 0.27→0.79，改善最剧烈 |
| credit-g | RF | 0.4988 → 0.5946 | 0.9570 → **0.6040** | **cost 全场新最优**（较 plain 最优 0.884 再降 32%） |
| credit-g | LR | 0.5307 → 0.5908 | 0.8840 → 0.6350 | 冒烟点，改善符合预期 |
| credit-g | SVC | 0.4979 → 0.5903 | 0.9600 → 0.6870 | — |
| sick | DT | 0.8362 → 0.8203 | —（sick 无 cost） | **唯一下降**（-0.016）：τ 按内层 OOF F1 选，DT 概率在 sick 上内层/外层分布略漂移 |
| sick | RF | 0.8703 → **0.8766** | — | 略升，holdout 0.9008 |
| sick | LR | 0.6867 → 0.7453 | — | +0.059 |
| sick | SVC | 0.7625 → 0.8087 | — | +0.046 |

初步观察（供 D3 / 报告，非结论）：threshold 在 credit-g 上以 accuracy 换 recall/f1/cost（acc 降 4–8 个点，cost 降 28–47%），与 5:1 代价矩阵的业务目标一致；sick 上 4 模型 3 升 1 微降。两协议方向一致（credit-g cost 排序、sick f1 排序均不翻转）。

## 4. 锚点核对

| 锚点 | 预期 | 实测 | 结果 |
|---|---|---|---|
| **auc 与 plain 逐位相等**（8 配置 × 2 协议 = 16 组） | threshold 的最终重拟合 = plain 同款流水线同种子同数据 → 概率相同 → auc 必须逐位相等；不等即包装器泄漏/透传有 bug | 16/16 全部 `==` 为 True | ✅ |
| τ 网格合法性 | 全部 τ ∈ {0.05,…,0.95} 且为 0.05 整数倍 | 全部通过（实际取值 0.20–0.60） | ✅ |
| τ 方向合理性 | 少数类正类 → 最优 τ 应普遍 < 0.5 | 全部 ≤ 0.60，绝大多数 ≤ 0.40 | ✅ |
| ZeroR 链路 | 沿用 C3 已核对锚点（评估链路未改动） | evaluate.py 未改动、self_test 先行 | ✅ |
| 冒烟先行 | 单点（creditg×LR）秒级验证后再放全量 | 通过后才放 --matrix | ✅ |

## 5. 落盘自查（11 项，脚本化，PASS）

1. 16 组 folds+summary 成对齐全，命名与 C3/D2 同款（holdout 带 `_holdout` 中缀）；
2. folds 列 = fold_template **29 列**、summary 列 = template **49 列**，零偏差（reindex 保证）；
3. cv10 行数=10 且 n_folds=10；holdout 行数=1 且 n_folds=1；
4. holdout 全部 `*_std` 列为空，无 0 冒充；
5. 全部 `data_sha256` = split_manifest.json 现值；
6. credit-g 行 cost_mean/cost_pooled 有限；sick 行两列均空；
7. 每折 notes 含 `tau=` 与 `oof_f1=`（cv10 每组 10/10，holdout 1/1）；
8. preds 16 份行数全对（cv10=1000/3772，holdout=300/1132），列同 C3；
9. auc 与 plain 逐位相等（见第 4 节）；
10. 模板与切分索引未被扰动（模板仍空表头）；
11. threshold 相关新文件共 48 份（32 结果 + 16 preds），无覆盖既有文件。

## 6. 设计决定（C 单方面裁量的部分，请 D/A/B 过目）

1. **τ 记录在 `notes` 列**：fold_template 无 τ 专用列，为保列零偏差把 `tau=…;oof_f1=…` 写入唯一自由文本列 notes。若 D 认为 τ 应有专列，请给列名，C 改模板后重跑（成本：一次全量）；
2. **threshold 基座 = plain**：规格策略为四选一（plain/smote/cost/threshold），不做 threshold×smote 组合——组合属新配置，须规格单先扩；
3. **并列 τ 取最小**：网格按升序遍历、严格大于才替换 → 并列时保留最小 τ，确定性可复现；
4. **内层 F1 用 `sklearn.metrics.f1_score`**（与 evaluate.py 同一库函数，显式映射正类=1 后调用，不自写公式）；报告指标仍一律走 `compute_metrics`；
5. **勘误**：C3 记录与 run_experiment 旧注释写「fold_template 28 列 / template 46 列」，实测 **29 / 49** 列（列零偏差本身始终成立），本次一并修正。

## 7. 防泄漏自查（C4 特有风险点）

- τ 的选择信息全部来自外层训练折内部（内层 5 折 OOF），**未接触外层验证折**（D1 红线）；
- 内层每折 `make_base()` 造全新子流水线：插补/编码/缩放随内层折重拟合，杜绝「pre 先在全训练折拟好再做内层 CV」；
- 每个外层折独立选 τ（10 折 10 个 τ）——τ 是折内决策，不是全局超参；holdout 在 train_idx 上独立选 τ 后锁定；
- `auc == plain` 逐位相等是「最终模型与 plain 同构、只改判标签」的机器证明。

## 8. D1 确认清单 —— C 部分（C4 增量）

- [x] 阈值在训练折内层选择并锁定，未用外层验证标签（D1 cv10 第 5 条）
- [x] 每折全新实例（含内层每折全新子流水线）
- [x] 指标一律 evaluate.py；τ 选择用同库 f1_score，不自写公式
- [x] 输出列零偏差（29/49）；混装禁止（一个 records 一个配置一个协议）
- [ ] τ 记录位置（notes 列）待 D 确认（见 6.1）
