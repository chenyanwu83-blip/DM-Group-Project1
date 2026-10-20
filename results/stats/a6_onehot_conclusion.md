# A6 对照实验①结论：OneHotEncoder 有/无（LogisticRegression, credit-g）

设置：编码② OneHot(handle_unknown='ignore')+StandardScaler vs 编码③ Ordinal+数值原样；
LogisticRegression(max_iter=1000, random_state=42)；分层 10 折 CV(shuffle, seed=42)；正类 bad。
切分自检：与 D 锁定折表 cv_folds_credit-g.csv 逐折一致（同配方 StratifiedKFold(10, shuffle, seed=42)）。
数据：results/a6_onehot_comparison.csv（汇总）+ a6_onehot_folds.csv（折级明细）。

| 指标 | OneHot ② | Ordinal ③ | Δ(②−③) | 配对 t 检验 p（未校正） |
|---|---|---|---|---|
| Accuracy | 0.7520±0.0290 | 0.7180±0.0496 | +0.0340 | 0.101 |
| F1(bad) | 0.5307±0.0627 | 0.4057±0.1252 | +0.1250 | 0.021 |
| ROC-AUC | 0.7846±0.0412 | 0.7379±0.0556 | +0.0467 | 0.008 |

变换后特征维度：OneHot 61 维 vs Ordinal 20 维。
补充归因组 A6.3（序数+标准化，非冻结配置）：acc 0.7220±0.0439 /
F1 0.4150±0.1101 / AUC 0.7371±0.0550。

## 结论（4 句）

1. One-hot 编码在三项指标上全面优于序数编码：ΔAccuracy +0.0340、ΔF1 +0.1250、ΔAUC +0.0467（折级配对 t 检验 p：acc 0.101 / f1 0.021 / auc 0.008，未校正，n=10）。另，序数组（数值未缩放，credit_amount 量级达 10⁴）lbfgs 达到 max_iter=1000 未完全收敛（n_iter=1000），独热组 55 轮即收敛——不稳定优化本身也是 ③ 分支不适配线性模型的佐证。
2. 归因：补充组 A6.3（序数+标准化，F1=0.4150）仍靠近序数组（0.4057）而远离独热组（0.5307），说明 F1 差距主要来自**编码方式**（虚假顺序关系），而非数值缩放。
3. 对 RQ4 的含义：混合属性数据集上"编码方式与模型族匹配"是必要决策——线性/核模型走 OneHot+标准化（规格单②），树模型直接吃序数编码（规格单③），两分支已在 A5 规格单固化进主矩阵流水线。
4. 与主矩阵衔接：本对照属 plain 策略下的编码消融，A7 的 16 配置沿用规格单②③分支，不再改动。

## 报告段落（EN，供 sec3a_creditg.tex）

To quantify the effect of encoding (RQ4), we ran LogisticRegression under two preprocessing
branches of our frozen specification: (a) one-hot expansion of the 13 nominal attributes with
standardised numerics (61 features), and (b) ordinal encoding with raw numerics
(20 features), both evaluated by stratified 10-fold cross-validation
(seed = 42, positive class = bad). One-hot yielded accuracy 0.752, F1
0.531 and AUC 0.785, against 0.718 / 0.406 /
0.738 for the ordinal branch (fold-level paired t-test: F1 p=0.021,
AUC p=0.008). A supplementary control (ordinal encoding with standardised numerics)
shows that the gap is mainly attributable to the encoding itself rather than to scaling: ordinal
codes impose an artificial order on nominal attributes, which distorts the linear decision
boundary. One-hot encoding is therefore retained for all linear/kernel models in the main
experiment matrix, while tree-based models keep raw ordinal codes.

—— A 角色 2026-10-17 生成；D 请按 D1 口径复核指标方向（正类 bad、概率列方向）。
