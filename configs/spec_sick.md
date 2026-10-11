## 0. 统一常量

| 项 | 值 |
|---|---|
| random_state / seed | 42（全组锁死） |
| 正类 pos_label | 'sick'（D1 协议） |
| 读取方式 | `pd.read_csv(path, na_values='?')`（'?' 统一转 NaN） |
| 评估 | StratifiedKFold(10, shuffle=True, random_state=42) + 70:30 分层留出（random_state=42，行索引存 results/holdout_idx.csv） |

## 1. 预处理组件

| # | 组件 | 参数 | 理由 |
|---|---|---|---|
| ① | 删全缺列 | `df.drop(columns=['TBG','TBG_measured'])`，在 Pipeline 外 | TBG 100% 缺失（该化验从未做过，配对标记 TBG_measured 恒为 'f'）——无任何观测值，插补=编造数据。列删除不依赖统计量，无泄漏风险 |
| ② | 缺失插补 | 数值 `SimpleImputer(strategy='mean')`；名义 `SimpleImputer(strategy='most_frequent')`，**在 Pipeline 内** | 缺失机制为"该项化验没做"（MAR）。B4 实验证明：删行损失 29.9% 样本且 F1 无统计优势；插补保留全部数据、结果最稳。折内拟合防泄漏 |
| ③ | 树模型编码 | `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)` | 树模型按分裂点切分，序数编码足够；one-hot 反而稀释分裂增益 |
| ④ | 线性模型预处理 | 数值 `StandardScaler()` + 名义 `OneHotEncoder(handle_unknown='ignore')` | Logistic/SVC 基于距离与权重，需统一量纲 + 类别数值化 |
| ⑤ | SMOTE | `SMOTE(random_state=42)`，仅在策略=smote 时加入 Pipeline | 仅在训练折合成少数类（imblearn Pipeline 保证）；验证折分布保持真实 |

## 2. Pipeline 结构

```
树模型:  pre(②插补+③序数) → [smote] → clf(DecisionTree/RandomForest)
线性模型: pre(②插补+④标准化/独热) → [smote] → clf(Logistic/SVC)
```

## 3. 模型与策略固定超参

| 模型 | 超参 |
|---|---|
| DecisionTree（原 J48） | criterion='entropy', ccp_alpha=0.01, random_state=42 |
| RandomForest | n_estimators=100, random_state=42, n_jobs=-1 |
| Logistic | max_iter=1000, random_state=42 |
| SVC（原 SMO） | kernel='rbf', C=1.0, probability=True, random_state=42 |

| 策略 | 实现 | sick 侧说明 |
|---|---|---|
| plain | 无附加 | 基线 |
| smote | Pipeline 内 SMOTE(random_state=42) | sick 侧主策略（重采样为主） |
| cost | class_weight={'sick':1,'negative':1} | **等代价对照**：官方 5:1 代价矩阵只属于 credit-g；等代价结果应≈plain，用作 sanity check |
| threshold | 训练折内 5 折 OOF 概率选 τ（0.05–0.95，步长 0.05，最大化 F1）→ 锁定 → 全训练折重拟合 → 按 τ 判 | 阈值在测试前锁定，防泄漏 |

## 4. B4 对照实验结论

删整行损失 29.9% 样本（3,772→2,643，正类 231→212）后 F1=0.843±0.069，
折内插补 F1=0.830±0.049（AUC 0.974 vs 0.964）：F1 差异在 1 个标准差内，
插补保数据全、方差小、AUC 高 → 采用插补。详见 results/b4_missing_comparison.csv。
