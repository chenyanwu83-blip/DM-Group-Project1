# D Report sec6+sec7 · v2.2 压缩稿（2026-11-02）

**基线** = `D_Report_Complete_Tables_20261102.docx`。数字全部逐字保留（已对 `results/all_metrics_v1.csv`、`b4_missing_comparison.csv`、tex v2.1 的数字复核），只动结构、句式与限定语。正文（不含表格图注）1,572 词 → 约 1,340 词（−15%）；页数大头在表，正文已到「不减难度」的下限。

**本版改动**（相对 Complete_Tables 版）：
1. 每条要点改为「粗体断言开头 + 最多一句限定」；删 6.1 重复对冲句、6.2 第 4 条（未定义的 accuracy–recall rule，ZeroR 已在 6.1 讲过）、6.2 第 3 条并入第 1 条尾句
2. RQ2 落点写进开头路标句；6.1 第 3 条重写（原三件事混装 + "Figure AUC" 语法错误）
3. 6.3 第 4 条改判词：deletion F1 更高，结论改为「三臂 <0.02 F1 → 保留记录 + AUC 最优」；补 B4 记录归属（决策树 + 自带编码包装，基线刻意不同于表 6.2 的 DT 行）
4. 黑话清理：archived→recorded/pre-specified/fixed；120 对家族加构成说明（16 配置两两成对）
5. 7.1 四条 finding 全带数字；6.3 补回「赢家翻转 = 四因素共同作用」（D6 核心结论）
6. ZeroR 正文 4 处→2 处（6.1、6.4）；116 秒独立成句；删 "confidence, not lift" 防御句；删规则选择准则细节保留一行括号版
7. 与 tex 系 v2.2 的 [D-a][D-b][D-c] 对齐：陷阱段口径（acc 0.71–0.77 高于 ZeroR 的 0.700、recall 0.27–0.40）、SMOTE 分通道插值表述（线性通道 one-hot、树通道序数编码，证据 b5_print_pipeline.py）

**仍开放（非文字问题，等组长/E 拍板）**：
- [ ] 6.1/6.2 是否合并——待组长/E 确认；确认后 Table 6.3 与其两条并入 6.1，后续小节顺移
- [ ] 图 6.2（策略增益图 fig_d_strategy_gain）是否整体删除——本稿暂保留与 docx 版一致；若删，需同步删 6.2 第 2 条对它的引用，且课程 ≥4 图红线使 A4 变为必做
- [ ] 图 6.3 本体重生成（`src/d5_figures.py` 轴向修正），caption 已按「行=实际，列=预测」写
- [ ] A4 主结果分组条形图未落（若 Figure 6.2 删除则从「建议」升级为「必须」）

---

# 6 Results and Evaluation

We evaluate four classifiers — decision tree, random forest, logistic regression and SVC — under plain training and three imbalance strategies: SMOTE, cost-sensitive weighting and threshold adjustment, with the preprocessing of Section 3 and the protocols of Section 5. The positive class is bad for credit-g and sick for sick. Sections 6.1–6.2 report the main matrices and statistical evidence (RQ1), the ZeroR references showing when accuracy misleads (RQ2); Section 6.3 adds cross-dataset and preprocessing evidence (RQ3); Section 6.4 decision quality on the holdout (RQ4); Section 6.5 the exploratory extensions.

## 6.1 Main Results across Two Datasets

**Table 6.1** credit-g F1 under ten-fold cross-validation

| Model | Plain | SMOTE | Cost-sensitive | Threshold |
|---|---|---|---|---|
| Decision Tree | 0.322 ± 0.192 | 0.471 ± 0.078 | 0.523 ± 0.030 | 0.561 ± 0.038 |
| Random Forest | 0.499 ± 0.079 | 0.509 ± 0.077 | 0.595 ± 0.064 | 0.595 ± 0.059 |
| Logistic Regression | 0.531 ± 0.060 | 0.591 ± 0.051 | **0.597 ± 0.034** | 0.591 ± 0.049 |
| SVC | 0.498 ± 0.101 | 0.560 ± 0.050 | 0.593 ± 0.032 | 0.590 ± 0.055 |
| ZeroR reference | F1 = 0; accuracy = 0.700; recall = 0 ||||

**Table 6.2** sick F1 under ten-fold cross-validation

| Model | Plain | SMOTE | Cost-sensitive | Threshold |
|---|---|---|---|---|
| Decision Tree | 0.836 ± 0.050 | 0.747 ± 0.050 | 0.836 ± 0.050 | 0.820 ± 0.048 |
| Random Forest | 0.870 ± 0.047 | **0.897 ± 0.051** | 0.873 ± 0.048 | 0.877 ± 0.052 |
| Logistic Regression | 0.687 ± 0.091 | 0.521 ± 0.028 | 0.687 ± 0.091 | 0.745 ± 0.038 |
| SVC | 0.763 ± 0.064 | 0.742 ± 0.039 | 0.763 ± 0.064 | 0.809 ± 0.039 |
| ZeroR reference | F1 = 0; accuracy = 0.939; recall = 0 ||||

Positive-class F1, mean ± population SD over ten folds; bold marks the best mean per matrix. The sick cost arm is an equal-weight control — sick has no official cost matrix — reproducing plain training for DT, LR and SVC; credit-g uses the official 5:1 weights. Source: results/all_metrics_v1.csv.

1. **The winners differ, and so do the interventions.** Cost-sensitive logistic regression leads credit-g (F1 0.597); random forest with SMOTE leads sick (0.897). On credit-g every intervention raises F1 for every model; on sick, SMOTE lowers F1 for three of four models while pushing all their recalls above 0.87 — the gain is bought with precision.

2. **Accuracy is cheap; minority detection is not.** The ZeroR references reach 0.700 and 0.939 accuracy while detecting no positive case, and three of four plain credit-g models hold accuracy 0.71–0.77 at recall 0.27–0.40 (RQ2). Minority performance must be read from precision, recall and F1 directly.

3. **Ranking quality is strategy-invariant on sick.** In Figure 6.1 the three random-forest ROC and PR curves nearly coincide (AUC 0.997–0.998): the interventions move the operating point, not the score ranking, and threshold adjustment shares the plain model's scores by construction. Curves pool the 3,772 out-of-fold predictions; the tables average folds.

**Figure 6.1** ROC and precision–recall curves for random forest on sick under plain, SMOTE and equal-cost training. Scores pooled across all 3,772 out-of-fold predictions; positive = sick; dotted lines mark chance and prevalence. Source: results/preds/sick_rf_*_v1_run1_cv10_preds.csv.

## 6.2 Strategy Effects and Statistical Evidence

Figure 6.2 shows mean changes against plain training; Table 6.3 separates descriptive gains from statistical evidence.

**Figure 6.2** Change in F1 and recall relative to each model's plain configuration, averaged over the four classifiers under ten-fold cross-validation. Descriptive changes, not significance. Source: results/cross_dataset_comparison_v1.csv.

**Table 6.3** Significant strategy-versus-plain contrasts at p < 0.05

| Dataset | Metric | Contrasts | Raw p | Holm p |
|---|---|---|---|---|
| credit-g | F1 | 12 | 3 | 0 |
| sick | F1 | 12 | 2 | 0 |
| credit-g | Recall | 12 | 11 | 8 |
| sick | Recall | 12 | 7 | 2 |

Fold-level paired t-tests with the Nadeau–Bengio overlap correction; Holm adjustment over the pre-specified family of all 120 pairwise contrasts among the 16 model–strategy configurations per dataset and metric — not recomputed over the twelve strategy-versus-plain contrasts alone. Source: results/ttest/*_ttest_pairs_{f1,recall}_v1.csv.

1. **The strategies reliably buy recall; F1 gains depend on the dataset.** Recall rises in 21 of 24 contrasts and never falls — 18 raw-significant, 10 after Holm; F1 passes the raw threshold in 5 of 24 and none survives Holm. A failed contrast is unproven, not absent.

2. **Recall gains can be offset by false positives.** Four sick configurations lose F1: SMOTE with decision tree, logistic regression and SVC, and threshold with decision tree — losses that Figure 6.2's model averages hide.

## 6.3 Cross-Dataset Comparison and Preprocessing Evidence

Figure 6.3 contrasts the error profiles of the two F1 winners on pooled cross-validation predictions — distinct from the holdout of Section 6.4.

**Figure 6.3** Pooled ten-fold confusion matrices. Top: credit-g logistic regression with cost weighting; bottom: sick random forest with SMOTE. Rows are actual labels; columns are predictions; left counts, right proportions within each actual class. Source: results/all_metrics_v1.csv.

1. **Different tasks reward different error profiles.** The credit-g winner catches 259 of 300 bad records but flags 309 of 700 good ones; the sick winner detects 205 of 231 positives with 21 false alarms. Model choice should follow the task's objective; Section 6.4 prices both in decision terms. The winner flip traces jointly to dataset size, attribute composition, missingness and imbalance — no single factor was isolated.

2. **Encoding matters where the model is linear.** For credit-g logistic regression: one-hot F1 0.531, ordinal 0.406, standardised ordinal 0.415 — evidence for the chosen LR configuration, not a universal rule (Section 3).

3. **The three strategies intervene at different points** — training data (SMOTE), training objective (costs), decision rule (threshold) — so their effects need not transfer. Ordinary SMOTE interpolates in the encoded feature space — one-hot for the linear channel, ordinal codes for trees — and can synthesise fractional or unnatural category values: a plausible reading of the sick losses, not a mechanism our experiments isolate.

4. **Missing-data handling changes little on sick.** In the frozen three-arm ablation (decision tree, cv10), imputation keeps all 3,772 rows at F1 0.830 ± 0.049 and AUC 0.974; deleting the 1,129 incomplete rows — 19 of 231 positives among them — moves F1 to 0.843 ± 0.069 while AUC falls to 0.964; marker-value handling gives 0.828 and 0.965. With all arms within 0.02 F1, records are retained and imputed within folds: every positive label is preserved and AUC is best. The ablation's own encoding wrapper makes its baseline differ from Table 6.2's DT row by design; the arms are compared only among themselves.

## 6.4 Holdout Results and Decision Trade-offs

Table 6.4 applies the F1-selected configurations to the fixed 70:30 holdout. Full-data cross-validation informed the selection, so this is an internal check on unseen samples, not a fully independent test.

**Table 6.4** Holdout performance of the main-matrix F1 winners

| Dataset | Configuration | N / positives | Recall | Precision | F1 | Cost |
|---|---|---|---|---|---|---|
| credit-g | LR + cost | 300 / 90 | 0.822 | 0.448 | 0.580 | 0.570 |
| sick | RF + SMOTE | 1,132 / 69 | 0.957 | 0.868 | 0.910 | — |

Cost is the credit-g relative error cost, (5·FN + FP)/N; sick has no official cost matrix. Source: results/all_metrics_v1.csv, holdout rows.

1. **credit-g: the champion controls cost.** Cost-weighted LR identifies 74 of 90 bad applicants, misses 16, flags 91 good: cost 0.570, against 0.887 for plain LR and 1.500 for always-approve ZeroR — 36% and 62% lower.

2. **F1 selection is not cost selection.** Random forest with threshold adjustment posts a lower holdout cost (0.543): the F1-selected champion is not the cost-optimal model, and relative cost is not financial loss.

3. **sick: both error types stay small.** Random forest with SMOTE identifies 66 of 69 positives and raises 10 false alarms among 1,063 negatives — a small, reviewable pool.

4. **Scope.** One same-period holdout per dataset certifies internal decision quality, not clinical safety or deployment readiness; an isolated final test and external validation remain open.

## 6.5 Exploratory Ensembles and Association Rules

These extensions reuse the fixed splits and pipeline but stand outside the main statistical family.

**Table 6.5** Exploratory ensembles under ten-fold cross-validation

| Dataset | Configuration | F1 mean ± SD | Recall | AUC | Runtime s |
|---|---|---|---|---|---|
| credit-g | Bagging + plain | 0.510 ± 0.072 | 0.430 | 0.782 | 2.2 |
| credit-g | Bagging + SMOTE | 0.548 ± 0.074 | 0.560 | 0.770 | 1.2 |
| credit-g | AdaBoost + plain | 0.499 ± 0.070 | 0.473 | 0.735 | 1.2 |
| credit-g | AdaBoost + SMOTE | 0.501 ± 0.087 | 0.483 | 0.723 | 2.4 |
| sick | Bagging + plain | 0.819 ± 0.050 | 0.775 | 0.990 | 3.6 |
| sick | Bagging + SMOTE | 0.785 ± 0.037 | 0.987 | 0.994 | 4.6 |
| sick | AdaBoost + plain | 0.918 ± 0.045 | 0.901 | 0.999 | 9.5 |
| sick | AdaBoost + SMOTE | **0.937 ± 0.043** | 0.931 | 0.999 | 28.9 |

F1 for the positive class; bold marks the best exploratory mean per dataset; runtime is recorded fit time. Source: results/ensemble_metrics_v1.csv.

1. **Ensembles extend the sick ranking.** AdaBoost with SMOTE posts the highest exploratory F1 (0.937); its gap to the main-matrix winner RF + SMOTE (0.897) was not tested. On credit-g both families stay below cost-weighted LR (0.597).

2. **The pre-specified tests support one narrow claim.** Boosting as a resampling substitute wins on sick (AdaBoost plain vs decision tree + SMOTE, p = 0.00064); bagging versus the single tree is not significant (p = 0.067–0.071), nor is the credit-g boosting contrast (p = 0.389) — and these tests sit outside the Holm family.

3. **SMOTE does not spare every ensemble.** Bagging with SMOTE on sick trades F1 0.819 → 0.785 for recall 0.775 → 0.987. Fitting totals about 116 s across all 84 recorded runs, the dearest 28.9 s — the full matrix reproduces on a laptop in minutes.

4. **Rules add readable evidence.** The strongest rule — duration in its top bin with savings below 100 — covers 124 records, 71 of them bad: confidence 0.573 against the 0.300 base rate, lift 1.91; six rules passed the fixed criterion (confidence ≥ 0.50 and ≥ 5 points over the stronger single-condition rule). Duration, amount and account status recur in both rules and tree interpretations; Apriori reads all 1,000 records against the trees' 700-record training split, so the overlap is suggestive, not validation.

# 7 Conclusion and Future Work

## 7.1 Findings

Four conclusions, one per research question; each rests on Section 6's evidence.

1. **(RQ1) Model and strategy choice is dataset-dependent.** credit-g rewards linear, cost-sensitive learning — cost-weighted LR, F1 0.597 under the official 5:1 cost; sick rewards ensemble plus resampling — RF with SMOTE, F1 0.897. No intervention improves every model on both datasets, and recall gains survive correction far more often than F1 gains: 10 of 24 contrasts versus none.

2. **(RQ2) Accuracy misleads under imbalance.** ZeroR reaches 0.700 and 0.939 accuracy with zero recall; plain credit-g models hold accuracy 0.71–0.77 at recall 0.27–0.40. Decisions must be read from minority-class precision, recall and F1, plus the official cost where one exists, with ROC and PR curves as ranking evidence.

3. **(RQ3) Preprocessing conclusions must match their experiments.** One-hot wins for the tested LR configuration (0.531 vs 0.406 ordinal); on sick, imputation, deletion and marker handling sit within 0.02 F1 and records are retained. Neither result is universal, and the winner flip reflects size, attribute mix, missingness and imbalance jointly.

4. **(RQ4) Decision quality is quantified within the internal benchmark.** On the holdout, the credit-g champion cuts the official cost from 1.500 (always-approve) and 0.887 (plain LR) to 0.570 while catching 74 of 90 bad applicants; the sick champion catches 66 of 69 patients at precision 0.868. Selection overlapped with full-data cross-validation — an internal benchmark, not deployment certification; an isolated final test remains future work.

Within these bounds the project delivers a reproducible model–strategy comparison with quantified minority-class trade-offs on two datasets; the exploratory ensembles and rules of Section 6.5 mark what is most worth testing for stability next.
