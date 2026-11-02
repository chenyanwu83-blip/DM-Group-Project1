# RQ 定义与映射（2026-10-29 起生效）

这份是全组一起表决通过的（对着《Rubrics》的 11 项标准逐条过了一遍改出来的，主要是补上 LO7 "deliver learnt models" 那个缺口）。从 10-29 起生效：报告 sec2（A8 写）和 ch6/ch7（D7/E8）都用这份的定义。旧版原来写在 README 第 5 行（已经原地更新）；历史记录里出现的旧编号按 §2 的映射去读。

## 1. 新版研究问题（以这份为准）

| # | 研究问题 | 答案要点（数字均取自已入库结果） |
|---|---|---|
| **RQ1** | 在类别不平衡、代价不对称的约束下，四种备选策略（plain/SMOTE/代价敏感/阈值移动）与四类模型该如何选？ | 冠军随数据集翻转：credit-g（干净、2.3:1）→ **lr_cost**（F1 0.5966、cost 0.884→0.514）；sick（29.9% 行缺失、15.3:1）→ **rf_smote**（F1 0.8972、holdout 0.9103）。四策略均买到召回（ΔRecall 12 组全 ≥0），SMOTE 的 F1 净收益翻转（+0.070→−0.062，唯一倒退：LR 0.687→0.521，树系免疫）。翻转由规模/属性/缺失/不平衡四维合取解释。限制：单 seed；Holm 校正后 F1 族保守（credit-g 1 对）、召回增益稳健（8/12）。 |
| **RQ2** | accuracy 何时失效？应按什么指标做评估与决策？ | 多数类占比即 accuracy 地板（ZeroR 0.70/0.94），预注册规则（acc≥0.70 且 recall≤0.40）命中 10 单元；决策指标由业务定——credit-g 用官方 5:1 代价口径、sick 用正类召回；AUC 独立度量判别力且策略不改判别力（threshold 与 plain 的 AUC 16/16 逐位相等，三策略 ROC 重合 0.997）。限制：AUC 极不平衡下可乐观，并用 PR 曲线。 |
| **RQ3** | 混合属性如何同时决定特征处理与模型选择（哪些特征重要、怎么编码）？ | 编码与模型咬合：线性模型必须 OneHot（F1 +0.125、p=0.021，收敛 61 次 vs 1000 次不收敛），树吃 Ordinal。特征重要性三法互证：RF top3（credit_amount 0.141/duration 0.107/checking_status 0.107）与 Apriori 最硬规则"时长最高档+低储蓄→坏账"（置信 57.3%、lift 1.91）字段级吻合；树路径未复现双条件规则——证据互补、关联非因果。 |
| **RQ4** | 学到的模型对新样本能做出可靠决策吗？（对应 LO7） | holdout 未见样本实测：credit-g 300 名新申请者——lr_cost 抓住 82% 坏客户（recall 0.822）、业务代价 0.57（ZeroR 1.5、plain 最优 0.88）；sick 1,132 名新患者——rf_smote 抓住 95.7% 病人（recall 0.957、precision 0.868）；Apriori 规则给出可读决策理由（"时长>30 月且储蓄<100 → 建议人工复核"）。限制：holdout 单次评估、同分布，上线需监控漂移。 |

## 2. 旧版 → 新版映射（看历史文档的时候按这个对照）

| 旧 RQ（V3–V5 方案时期） | 去向 | 解读规则 |
|---|---|---|
| 旧 RQ1 策略相对效果随不平衡度变化 | → **新 RQ1** | 主题不变（策略选择）；历史"RQ1 素材"读作新 RQ1 证据 |
| 旧 RQ2 accuracy 失效与指标选择 | → **新 RQ2** | **语义与编号均不变**；代码常量 `RQ2_ACC_MIN/RQ2_RECALL_MAX` 与预注册规则继续有效 |
| 旧 RQ3 官方 5:1 代价矩阵下最优模型是否改变 | → **并入新 RQ1**（"代价不对称约束"与冠军结论）；其 holdout 决策面同时支撑新 RQ4 | 历史"RQ3 素材"（cost 组 recall 大涨、cost 0.884→0.514 等）读作新 RQ1 的代价维度证据 |
| 旧 RQ4 混合属性对预处理与模型选择的影响 | → **新 RQ3** | 仅编号迁移；A6 记录/结论文件中的"RQ4"一律读作新 RQ3 |
| （无） | **新 RQ4 为全新问题** | 由既有 holdout 结果回答，无需补实验（零新实验，符合冻结令） |

## 3. 逐文件对照清单（把仓库里的 RQ 引用过了一遍 @ fcca720）

| 文件 | 引用内容 | 按新版应读作 | 是否改文件 |
|---|---|---|---|
| docs/A7_creditg_smote_cost运行记录_v1.0.md | RQ3/RQ2/RQ1 素材各一处 | RQ3→新RQ1；RQ1/RQ2 不变 | ✅ 已追加尾注 |
| docs/A7x_ensemble_creditg运行记录_v1.0.md | "RQ4 的完整故事"（指 A6 编码） | 新 RQ3 | ✅ 已追加尾注 |
| docs/C5_统计检验运行记录_v1.0.md | RQ2/RQ3 素材 | RQ3→新RQ1 | ✅ 已追加尾注 |
| docs/B10_讲稿与幻灯要点_v1.0.md | "支撑 RQ1–4" | 新 RQ1–4 | ✅ 已追加尾注 |
| docs/D3、D4、D6、D7_sec6 draft、C3 | 仅 RQ2（accuracy 陷阱/分歧） | 新 RQ2（语义不变） | 不改，本行即为指引 |
| src/d4_all_metrics.py、d3_plain_metrics.py、stats_test.py、a4_creditg_eda.py | 代码内 RQ2 标志/注释 | 新 RQ2（不变） | **不改**（冻结令） |
| src/a6_creditg_onehot_contrast.py、results/stats/a6_onehot_conclusion.md | "RQ4"（编码问题） | 新 RQ3 | **不改**（冻结令/结果文件），按本表解读 |
| configs/spec_creditg.md、src/e_apriori/a5_source_spec_creditg.md | "RQ3 主证据/RQ4 侧证据" | RQ3→新RQ1；RQ4→新RQ3 | **不改**（冻结规格），按本表解读 |
| figures/credit-g_fig_captions.md | RQ2 下限参照 | 新 RQ2（不变） | 不改（生成物） |

## 4. 冻结纪律

按 M3 冻结令（2026-10-23），src/、configs/、results/、figures/ 内已入库文件一律不因 RQ 改编号而修改——旧编号一律以本文件 §2/§3 为准解读。本文件与 README L5 为 RQ 的唯一现行定义处；线上 Overleaf main.tex 的 sec2 目前仍为模板骨架，A8 撰写 sec2 时直接引用本文件 §1 的新四条。
