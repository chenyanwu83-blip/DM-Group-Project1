# D 复核记录：A6 对照实验①指标口径（v1.0）

| 项 | 内容 |
|---|---|
| 复核人 | D（评估与比较）——分工方案 A6 协作栏"D 审指标口径" |
| 日期 | 2026-10-20 |
| 被复核物 | `src/a6_creditg_onehot_contrast.py`、`results/a6_onehot_comparison.csv`、`results/a6_onehot_folds.csv`、`results/stats/a6_onehot_conclusion.md`（A 于 10-17 交付） |
| 复核方式 | 从折级明细独立复算全部汇总数字与检验；与 C3 同配置结果交叉验证；独立重建折切分比对锁定折表 |
| **结论** | **通过**。指标口径与 D1 协议一致，结论段全部数字可复算复现。一条 std 口径提示见 §3 |

## 1. 口径核查项

| 核查项 | 结果 | 依据 |
|---|---|---|
| 正类口径 = bad | ✅ | `y=(class=='bad').astype(int)`，sklearn 默认 pos_label=1 ⇒ 即 bad；F1/AUC 均对正类 bad 计算 |
| AUC 用概率而非标签 | ✅ | `cross_validate(scoring='roc_auc')` 走 `predict_proba`；且与 C3 概率版 AUC 逐位一致（§2） |
| 折切分 = D 锁定折表 | ✅ | D 独立重建 `StratifiedKFold(10, shuffle, random_state=42)` 与 `cv_folds_credit-g.csv` **逐折一致**（不采信 A 的自检输出，独立重跑） |
| 检验方法定位 | ✅ | 折级配对 t 检验明确标注"未校正，n=10，仅作差异概览；Nadeau & Bengio 校正由 C5 负责"——与 D1/方案口径不冲突 |
| 防泄漏 | ✅ | 正式评估全部在 `cross_validate` 折内拟合；全量 fit 仅用于数特征维度与记录 n_iter，不产出指标 |

## 2. 数字复算（与 a6_onehot_conclusion.md 逐一对照）

| 指标 | OneHot ② | Ordinal ③ | Δ(②−③) | p（复算） | p（结论表） |
|---|---|---|---|---|---|
| Accuracy | 0.7520 | 0.7180 | +0.0340 | 0.101 | 0.101 |
| F1(bad) | 0.5307 | 0.4057 | +0.1250 | 0.021 | 0.021 |
| ROC-AUC | 0.7846 | 0.7379 | +0.0467 | 0.008 | 0.008 |

均值、差值、p 值全部复算一致。归因组 A6.3（序数+标准化）均值 0.7220/0.4150/0.7371 亦复算一致，"差距来自编码而非缩放"的归因方向成立（0.4150 靠近 0.4057、远离 0.5307）。

**交叉验证（机器级证据）**：A6 的 OneHot 组与 C3 的 `creditg_lr_plain_v1`（Logistic + OneHot + StandardScaler，同折表）三个指标的**逐折值最大差异 = 0.00e+00**。两条独立产线得到逐位相同的结果，等价于证明 A6 的正类定义、切分与 AUC 概率口径与 D2/C3 主口径完全一致。

## 3. 一条口径提示（不改 A6 存档，写给后续写作）

A6 汇总表的 ±std 用的是 **ddof=1（样本标准差）**（如 F1 0.5307±0.0627），而 D1/D2 主矩阵口径为 **ddof=0（总体标准差）**（C3/D3 表中同配置报 0.5307±0.0595）。均值与检验不受影响，但**报告写作引用该配置 std 时请统一用主矩阵的 ddof=0 数值**（或显式标注 A6 为 ddof=1），否则 D8 数字关卡会把 0.0627 vs 0.0595 判为数字不一致。已在 D3 侧登记，A 无需改动任何已交付文件。

## 4. 复现

复核于 2026-10-20 以仓库根目录内联脚本执行：读 `results/a6_onehot_folds.csv` 复算均值/`scipy.stats.ttest_rel`；`pd.read_csv('results/creditg_lr_plain_v1_run1_folds.csv')` 逐折比对；`StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` 重建切分比对 `results/cv_folds_credit-g.csv`。环境：仓库 `.venv`（pandas 2.3.3 / scikit-learn 1.6.1 / scipy 1.13.1）。
