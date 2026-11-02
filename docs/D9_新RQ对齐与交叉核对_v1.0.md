# D9 新 RQ 对齐改稿与跨组员数字交叉核对 v1.0

- **负责人**: D（评估侧）
- **日期**: 2026-11-01
- **触发**: 全组决议（2026-10-29）新版四条 RQ 生效（commit 4579859, `docs/RQ_定义与映射_v1.0.md`），其中新 RQ4（新样本决策，LO7）官方落点 = ch6 + ch7 = D 的 sec6/sec7
- **交付物**: `docs/D7_sec6_results_draft_v2.0.tex`、`docs/D7_sec7_conclusion_draft_v2.0.tex`（本清单为其对账附件）
- **冻结令合规**: 零新实验、零数字重算——全部数字取自已入库冻结资产，仅做只读核对与文字重排

---

## 1. v1.0 → v2.0 改动摘要

| # | 改动 | 性质 |
|---|------|------|
| 1 | 各小节标题/首句挂 RQ1–RQ4 标签；6.1 末加 RQ→小节映射句 | 重排 |
| 2 | **6.3 sick cost 段机制勘误**: v1.0"概率太极端翻不动"解释作废 → 等代价对照（冻结权重 {1,1}）设计使然 + RF 平票翻转 | **实质勘误**（见 §3.4） |
| 3 | 新增 6.7 Decision Quality on Unseen Samples（RQ4），两数据集 holdout | 新增小节 |
| 4 | 6.5 属性构成段补 RQ3 交叉引用（A6 编码消融 + RF 重要性 + Apriori） | 补引用 |
| 5 | 6.4 补 RQ2 预注册规则命中（10 单元，实测 rq2_flag 列） | 补证据 |
| 6 | sec7: Findings 3→4 条（逐条挂 RQ）、Limitations 3→4 条（RQ4 单次评估/漂移）、给 E 的摘要注释 3→4 句 | 重排+新增 |

v1.0 中经 D8 全量对账的数字**零改动**（仅 6.3 机制句与 6.5/6.4/6.7 新增文字涉及新引用数字，全部在本清单 §2 逐项核对）。

## 2. v2.0 引用数字逐项核对表（本轮实测，2026-11-01）

### 2.1 新增 RQ4 小节（6.7）——全部来自 `results/all_metrics_v1.csv` holdout 行

| v2.0 文中数字 | 入库值 | 状态 |
|---|---|---|
| credit-g holdout 300 行（90 bad / 210 good） | n_eval=300; zero_ref: TN=210 FN=90 | ✅ |
| lr_cost: 74/90 bad（recall 0.822） | TP=74, FN=16, recall=0.8222 | ✅ |
| lr_cost: 错标 91/210 good | FP=91, TN=119 | ✅ |
| lr_cost: F1 0.580, cost 0.570 | f1=0.5805, cost_pooled=0.570 | ✅ |
| lr_plain（plain 最优）cost 0.886 | cost_pooled=0.8867 | ✅ |
| ZeroR（全批好）cost 1.500 | cost_pooled=1.500 | ✅ |
| 36%/62% 节省 | 0.570/0.886=0.643（↓35.7%）; 0.570/1.500=0.62 | ✅ |
| rf_threshold holdout cost 0.543（明示未预指定、仅描述） | cost_pooled=0.5433 | ✅ |
| sick holdout 1,132 行（69 sick） | n_eval=1132; TP=66+FN=3=69 | ✅ |
| rf_smote: 漏 3/69（recall 0.957）、10 假警/1063 健康（precision 0.868） | FN=3, TP=66, FP=10, TN=1053 | ✅ |
| rf_smote: F1 0.910（cv10 0.897±0.051 的 1 sd 内） | f1=0.9103; cv10 0.8972±0.0510 | ✅ |
| lr_cost cv10 0.597±0.034（1 sd 内） | 0.5966±0.0341 | ✅ |

### 2.2 6.3 sick cost 机制勘误句——冻结规格与运行记录

| 断言 | 来源 | 状态 |
|---|---|---|
| sick cost 臂 = 等代价对照 `class_weight={'sick':1,'negative':1}` | `configs/spec_sick.md` §3（B5 冻结）；`src/c_config.py:57`；`docs/C2_流水线规格回执_v1.0.md` §cost 行 | ✅ |
| dt/lr/svc 与 plain 逐位恒等（p=1） | `docs/C5_统计检验运行记录_v1.0.md` §4⑤："dt/lr/svc 逐位相同(p=1, identical)" | ✅ |
| rf_cost≠rf_plain：汇池恰差 1 例（FN 44→43） | all_metrics: rf_plain cv10 FN=44/TP=187; rf_cost FN=43/TP=188 | ✅ |
| "平票翻转（RF 已知敏感性），如实入检验不作差异解读" | C5 §4⑤ 原文（max fold Δ=0.0833） | ✅ |
| 官方 5:1 仅属 credit-g | spec_creditg（A5 冻结）；c_config.py:47 `{"bad":5,"good":1}` | ✅ |

### 2.3 6.4 RQ2 预注册规则句——实测 `rq2_flag` 列

命中恰 **10 单元** = 4 ZeroR 参照格（两数据集 × 两协议，`zero_ref_trap`）+ credit-g plain 8 格中 6 格（dt/rf/svc × cv10/holdout，`acc_high_recall_low`；rf cv10 recall=0.400 恰在边界 ≤0.40；lr plain recall 0.47/0.48 未命中）。sick 无 plain 命中（少数类近可分，RF AUC 0.998）。与映射文档 RQ2 答案要点"命中 10 单元"一致 ✅。

### 2.4 6.5 RQ3 交叉引用句——A6 与 E 侧资产

| v2.0 文中数字 | 来源 | 状态 |
|---|---|---|
| OneHot 61 维 / Ordinal 20 维 | `results/a6_onehot_comparison.csv` n_features=61/20 | ✅ |
| F1 0.531 vs 0.406（p=0.021） | 同上 f1 0.5307 vs 0.4057；a6_onehot_conclusion.md p=0.021 | ✅ |
| AUC 0.785 vs 0.738（p=0.008） | 0.7846 vs 0.7379 | ✅ |
| **序数 1000 轮未收敛 / 独热 55 轮收敛** | n_iter_full_fit=1000(converged=False) / **55**(True) | ✅（注意：映射文档误写"61 次"，见 §3.1） |
| 归因组（序数+标准化）F1 0.415 | A6.3 f1=0.4150 | ✅ |
| RF 重要性 top3: credit_amount 0.141 / duration 0.107 / checking_status 0.107 | `src/e_apriori/e4_rf_feature_importance.csv` 0.1409/0.1074/0.1066 | ✅ |
| Apriori 最硬规则: 时长最高档 ∧ 储蓄<100 → bad，置信 57.3%（基线 30%），lift 1.91 | `src/e_apriori/e3_selected_rules.csv` 第 1 行 confidence=0.5726, lift=1.9086（30% 基线 = 300/1000 官方口径） | ✅ |

### 2.5 与组员成稿的一致性抽查（既有 D8 结论 + 本轮复核）

| 组员/文件 | 交叠数字 | 状态 |
|---|---|---|
| A8 sec2（overleaf/sections/sec2_introduction.tex, 11/1 新版） | RQ4 表述"locked hold-out 300 / 1,132"；70:30 vs 94:6；6% | ✅ 一致 |
| A8 sec3a（sec:data） | 13 名义+7 数值；0 缺失；700:300 | ✅ 一致 |
| B8 sec3b 草稿 | 3,541:231；29.9% 行缺失；删行丢 19 个正类（=231→212，A9 亦核） | ✅ 一致 |
| C7 C_sec45_draft（11/1 修订版） | 700/300、2640/1132、3772×27（=30−TBG×2−Class, c_config n_features=27）、代价口径 (5FN+FP)/N、"equal-weight control" | ✅ 一致（v1.0 与 C 稿矛盾处已在 v2.0 按 C 侧修正，见 §3.4） |
| C5 检验记录 | 12 预指定对照方向/p 值抽查：dt_smote sick F1 −0.089(p=0.048)、lr_smote −0.166(p=0.002)、rf_smote +0.027(n.s.)、recall 增益范围 | ✅ 一致 |
| C11 集成检验 | 4 数：bag vs dt +0.188(p=0.071)/−0.018(p=0.067)；ada vs dt_smote +0.028(p=0.389)/**+0.171(p=6.39e-4)** | ✅ 一致 |
| D5/D8 图与表 | Table E1 八行抽查（F1/recall/AUC/runtime）逐格一致；图 D1–D3 取数文件未动 | ✅ 一致 |

## 3. 发现的不一致（需处理，均非 D 侧资产）

### 3.1 【建议必改】映射文档 RQ3 答案要点："收敛 61 次" 应为 "55 轮"
- `docs/RQ_定义与映射_v1.0.md` §1 RQ3："收敛 61 次 vs 1000 次不收敛"——**61 是特征维度（OneHot 61 维），收敛轮数实测 55**（a6_onehot_comparison.csv: n_iter_full_fit=55/1000）。A6 结论文档与 sec3a 消融段本身无误（"55 轮即收敛""61 维"分开写）；混淆发生在 RQ docx（副本新RQ4.docx）→ 映射文档转抄时。映射文档是 ch6/ch7 的"唯一现行定义处"，若 A/E 在 sec3/sec5 引用"61 次"会复制错误。**修法**: "61 维独热 55 轮收敛 vs 20 维序数 1000 轮未收敛"。v2.0 sec6 已按 55/61 正确书写。

### 3.2 【建议必改】sec2 贡献段残留旧缺失口径 "about 16%"
- `overleaf/sections/sec2_introduction.tex` 贡献 (i)："missingness (0 vs.\ about 16\%)"。全组修正口径（D8-E2 勘误、A9 复核、C7 草稿已于 11/1 修正为）= **5.4% cells / 29.9% rows**。11/1 提交换 RQ 块时此行未同步。owner: A/E。**修法**: "missingness (0 vs.\ 29.9\% of rows)"。

### 3.3 【低风险】映射文档 RQ4 答案要点 "plain 最优 0.88"
- 实测 lr_plain holdout cost = 0.8867，两位舍入应为 **0.89**（0.88 是截断）。v2.0 sec6 直接写 0.886 规避。

### 3.4 【已修】v1.0 sec6 与 C 稿关于 sick cost 臂的矛盾（D 侧自纠）
- v1.0："DT/LR/SVC 概率太极端、权重翻不动任何预测"——与冻结规格冲突（等权重对照根本没有"权重调整"可翻），且与 C_sec45"equal-weight control"表述矛盾。v2.0 已按 spec_sick(B5)/C2/C5 权威链改写：恒等=设计使然的 sanity check；RF 单例偏差=显式等权 sample_weight 下树分裂平票翻转（C5 已如实记录）。**教训入档**: 机制解释必须回溯冻结规格，不能从结果反推故事。

## 4. 写回流程（不变，沿 D7 撤回约定）

1. 用户核验 v2.0 图表与文字（尤其 6.3 机制句、6.7 新小节、sec7 Findings④/Limitations④）。
2. 写回前 PULL FIRST（overleaf 为 E 的网页端 git 镜像）。
3. `cp docs/D7_sec6_results_draft_v2.0.tex overleaf/sections/sec6_results.tex`（去注释头），sec7 同理；figures/fig_d_*.pdf 三份拷贝按 D7 §4 编译清单回填。
4. §3.1/§3.2 两项转 A/E 修订（转 A：sec2 缺失口径；转 E/全组：映射文档 61 次→55 轮）。
