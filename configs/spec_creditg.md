# credit-g 预处理规格单 v1.0（角色 A，2026-10-12 冻结）

> 冻结规则（M2）：此后改任何参数须全组同意并记版本号。格式与 configs/spec_sick.md 对齐，保证两数据集流水线同构。
> 数据质量依据：A3 全项通过（1000×21、700:300、0 缺失、0 重复，results/stats/credit-g_stats.md）。

## 0. 统一常量

| 项 | 值 |
|---|---|
| random_state / seed | 42（全组锁死） |
| 正类 pos_label | 'bad'（D1 协议） |
| 读取方式 | `pd.read_csv(path)`（无缺失，无需 na_values） |
| 评估 | StratifiedKFold(10, shuffle=True, random_state=42) + 70:30 分层留出（random_state=42，行索引存 results/holdout_idx_creditg.csv） |

## 1. 预处理组件

| # | 组件 | 参数 | 理由 |
|---|---|---|---|
| ① | 数据清洗 | **无**（原始数据直接入 Pipeline） | A3 已验证 0 缺失、0 重复行；credit-g 无需插补/删列（与 sick 侧最大差异） |
| ② | 线性模型预处理 | `ColumnTransformer`：名义列 `OneHotEncoder(handle_unknown='ignore')` + 数值列 `StandardScaler()`，**在 Pipeline 内** | Logistic/SVC 基于距离与权重，需统一量纲 + 类别数值化；折内拟合防泄漏 |
| ③ | 树模型编码 | 名义列 `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`；数值列 `remainder='passthrough'` 不缩放 | 与 sick 侧同构：树按分裂点切分，序数编码足够，one-hot 稀释分裂增益；数值缩放对树无意义 |
| ④ | 离散化（供 E 的 Apriori，不进分类流水线） | `KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')` 作用于 duration / credit_amount / age；名义列原样保留 | fig3 显示三属性均右偏（credit_amount 中位 2320 ≪ 最大 18424 DM），等频切分防高频箱塌缩 |

## 2. Pipeline 结构

```
树模型:   pre(③序数编码+数值passthrough) → [smote] → clf(DecisionTree/RandomForest)
线性模型: pre(②独热+标准化)             → [smote] → clf(Logistic/SVC)
离散事务表(E): ④等频5箱(duration/credit_amount/age) + 名义原样 → Apriori
```

## 3. 模型与策略固定超参（与 sick 侧统一）

| 模型 | 超参 |
|---|---|
| DecisionTree（原 J48） | criterion='entropy', ccp_alpha=0.01, random_state=42 |
| RandomForest | n_estimators=100, random_state=42, n_jobs=-1 |
| Logistic | max_iter=1000, random_state=42 |
| SVC（原 SMO） | kernel='rbf', C=1.0, probability=True, random_state=42 |

| 策略 | 实现 | credit-g 侧说明 |
|---|---|---|
| plain | 无附加 | 基线 |
| smote | Pipeline 内 `SMOTE(random_state=42)`，仅训练折 | 70:30 中度不平衡的重采样主对照 |
| cost | `class_weight={'bad':5,'good':1}` | **官方 5:1 代价矩阵**（数据集随附：FN 误判 bad→good 的代价是 FP 的 5 倍）；credit-g 专属，RQ3 主证据 |
| threshold | 训练折内 5 折 OOF 概率选 τ（0.05–0.95，步长 0.05，最大化 F1，pos_label='bad'）→ 锁定 → 全训练折重拟合 → 按 τ 判 | 阈值在测试前锁定，防泄漏 |

## 4. A6 对照实验设计（2026-10-17 已跑，结果回填）

- **问题**：OneHotEncoder 对 Logistic 是否必要（RQ4 的 credit-g 侧证据）
- **设置**：`LogisticRegression(max_iter=1000)` × {编码=② OneHot vs ③ Ordinal} × 分层 10 折 CV，`scoring=['accuracy','f1','roc_auc']`（pos_label='bad'，其余超参同上表）；另加补充归因组"序数+标准化"（非冻结配置，仅拆分编码/缩放效应）
- **结果**（results/a6_onehot_comparison.csv，折级明细 a6_onehot_folds.csv，结论段 stats/a6_onehot_conclusion.md）：

| 指标 | ② OneHot+Std（61 维） | ③ Ordinal 原样（20 维） | Δ(②−③) | 配对 t 检验 p |
|---|---|---|---|---|
| Accuracy | 0.7520±0.0290 | 0.7180±0.0496 | +0.0340 | 0.101 |
| F1(bad) | 0.5307±0.0627 | 0.4057±0.1252 | **+0.1250** | **0.021** |
| ROC-AUC | 0.7846±0.0412 | 0.7379±0.0556 | +0.0467 | **0.008** |

- **结论**：One-hot 三项指标全面占优（F1/AUC 显著，p<0.05）。补充组（序数+标准化，F1=0.4150）仍贴近序数组而远离独热组 → 差距主因是**编码方式**（序数编码制造虚假顺序）而非缩放；缩放仅解决收敛（序数组 n_iter=1000 未收敛 vs 独热组 55 / 补充组 41）。②③分支维持冻结，A7 主矩阵照此执行。切分经自检与 D 锁定折表 cv_folds_credit-g.csv 逐折一致。


## 附注：§1 ④ Apriori 数字档位列澄清（2026-10-12）

`installment_commitment`、`residence_since`、`existing_credits`、`num_dependents` 保留原有整数档位，按 `column=value` 生成交易项（例如 `installment_commitment=4`）；不再分箱，也不排除。低支持度档位交由 Apriori 的 `min_support` 筛除。此澄清不改变 §1 ④ 对 `duration`、`credit_amount`、`age` 的五箱等频离散化参数。
