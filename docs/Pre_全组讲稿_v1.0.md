# Pre · 全组讲稿 v1.0（2026-10-29 · 新顺序 E→B→A→C→D→E 三明治）

> **使用规则**：① 每人可自由修改**自己段落**的内容与语气；② **衔接句（开头的接榫句 + 结尾的交棒句）是全场锚点，请勿单独改动**——要改请在群里统一确认，避免台上接不上；③ 所有数字来自锁定资产（D8 对账 0 FAIL），改数字前先对 `results/`；④ 全场 14′30″ + 30″ 转场 = 15′，**每人 ≥2 分钟是课程硬线**。
> 各段【讲法】为排练提示，不讲出。占位符：〔组名〕〔你的名字〕按实际替换。
> **语速校准**：全稿约 1,300 词，按 120 词/分钟的从容语速 + C 的 1.5′ demo + 转场停顿 ≈ 13′，对 15′ 留 2′ 缓冲——宁短勿超；每人可就自己幻灯片自然扩写 20–30 秒，但别加新论点。

## 总时间表

| # | 讲者 | 时长 | 内容 | 页数 |
|---|---|---|---|---|
| 1 | E | 2:00 | 开场：团队/分工/双数据集总览/RQ | 2 |
| 2 | B | 2:30 | sick 数据集 + 数据质量专题 + 集成彩蛋 | 3 |
| 3 | A | 2:30 | credit-g 数据集 + A6 编码对照实验 | 3 |
| 4 | C | 3:00 | 统一流水线 + 防泄漏 + 现场 demo | 3-4 |
| 5 | D | 3:00 | 结果三发现 + 跨数据集对比表 + Ensemble Study | 4 |
| 6 | E | 1:30 | 收尾：结论/局限/未来工作/引导 Q&A | 1-2 |

---

## 第 1 段 · E 开场（2:00）

> Good afternoon. We are 〔组名〕 — five people, one project, two datasets,
> and one big question: **what does class imbalance do to data mining?**
>
> Quick tour of the team: I integrated the report and ran the association
> analysis; Ling owned the sick dataset; hzy owned credit-g; cyw built our
> modeling pipeline; cyn designed the evaluation. Everyone you see will
> speak today.
>
> Our two datasets could not be more different. **credit-g**: one thousand
> loan applications from a German bank — small, clean, no missing values,
> 70 to 30 imbalanced. **sick**: three thousand seven hundred patient
> records from thyroid screening — nearly four times bigger, messy, with
> real missing values, and imbalanced **94 to 6**. Same toolbox, very
> different patients. That contrast is the whole point: size, imbalance,
> missingness, attribute types, and domain — five differences, which is
> exactly what the course asks for.
>
> So we ask four questions. One: do imbalance strategies — SMOTE,
> cost-sensitive learning, threshold moving — still work as imbalance gets
> extreme? Two: when exactly does accuracy lie to us? Three: does the
> official cost matrix change the best model? Four: how much does
> preprocessing change the ranking?
>
> Let's start with the messier one. **Ling will take you into sick.**

【讲法：报分工时手指到人；两个数据集那句左右手各指一侧幻灯片；四个问题快节奏一页带过，别展开；最后一句提高音量交棒】

---

## 第 2 段 · B（2:30 · LOCKED v1.6 用户最终版）

> Thanks, zyj. So — sick, from the UCI Thyroid Disease collection: 3,772 patient records, 30 attributes. And the problem jumps out on this slide: 3,541 negative against just 231 sick — roughly 94 to 6. There are two quick notes on wording, because we want to be precise: first, “sick” here means sick-euthyroid — normal thyroid function, but abnormal thyroid blood tests, usually from other serious illness — and it is NOT hypothyroidism. And “negative” just means the screening found nothing, not perfect health.

> Now, when I first opened this dataset, it looked really bad — every single row had missing values. But once I dug into why, everything changed — and this is the part worth remembering. Every lab test comes as a pair: a flag saying whether the test was done, and the value itself. So a false flag doesn’t mean we lost the value, it means the test was never done. The clearest case is TBG: 100% missing across all 3,772 rows. And I checked all six lab pairs — zero contradictions. So the dataset is incomplete, but it’s honest.

> And so that is how we handle it. You can not impute TBG — there’s nothing to impute from — so we remove it. For the rest, about 30% of rows still have gaps — so should we just delete those rows? I tested it: you’d throw away 30% of the patients, including 19 real sick patients, while F1 doesn’t even improve. Not worth it. So we fill the gaps instead — means for numbers, most-frequent for categories.

> And here’s the part people often miss: where you fill the gaps matters as much as how. Everything — imputation, encoding, scaling, SMOTE — lives inside the pipeline, fitted only on training folds, never on the test data. And because we all share the same locked splits, every number you’ll see today is directly comparable.

> One last surprise from our ensemble study: AdaBoost plus SMOTE reached F1 0.937, beating our best main-matrix model — re-weighting beat re-sampling. That’s the sick side. Now hzy will show you credit-g, and a clean experiment on why encoding matters.

【讲法】Slide 1 指右侧数字块；Slide 2 是全场最有戏的一段（先压低声音说“it looked really bad”，再抬起语气说“everything changed”）；Slide 3 反问句要像真在问观众；Slide 4 观点句抬头说；Slide 5 语气轻松，交棒词说清楚。

> **本段 338 词 · 2′30″–2′42″（125–135 词/分）· v1.6 LOCKED 逐字定稿，不再改动**

## 第 3 段 · A（2:30）

> Thanks, Ling. After all those missing values, let me show you something
> much cleaner: **credit-g** — one thousand loan applications from a German
> bank, from 1994, UCI ID 144. **1,000 rows, 21 columns**, mostly
> categorical, and **zero missing values, zero duplicates**. The class is
> 700 good versus 300 bad — 70 to 30. We treat **bad** as the positive
> class — the official cost matrix says missing a bad loan costs **five
> times** more than rejecting a good one.
>
> One thing stands out: **checking status**. On this plot, the four groups
> are sorted by default rate, and it goes down step by step — **people
> with an account below zero default 49% of the time; people with no
> checking account, only 12%**. The average is 30%, in the middle. Less
> money in the account, more risk — just what a bank would expect.
>
> Now, the experiment I ran. The question: with mixed data — numbers and
> categories — **does encoding matter?** I took the same logistic
> regression and gave it two versions of the same data. Version one:
> one-hot encoding plus scaling — 61 columns. Version two: the categories
> just turned into numbers — 20 columns. Same model, same folds, same
> seed — **only the encoding is different**.
>
> The result is clear: **F1 jumps from 0.41 to 0.53**, and AUC goes up
> as well — both gains are statistically significant. We also checked:
> the gap comes from **the encoding, not the scaling**. Why?
> When you turn categories into numbers, you create an order that does
> not exist — and a linear model believes that order. Tree models don't
> care, so they keep the simple numbers. So on mixed data, how you encode
> can change which model wins — that's our answer to **research question
> three**.
>
> So now you've seen both datasets — one messy, one clean. **C will show
> you how we model both with one shared pipeline.**

【讲法：本段与 docs/A10_讲稿与幻灯要点_v1.0.md 同源（A 认领修订 11-02：修正探索段方向、RQ 改新编号、去掉 duration 超卖；晚间改为短句口语版约 300 词）。49% 和 12% 两个数字对着幻灯上排好序的四根柱子点，**方向别说反——账户低于零（<0）最高、无账户最低**；0.41→0.53 用手势比出"跳"；**p 值数字不念**（幻灯 A3 表第三行有），只说 "both gains are statistically significant"，被问再报 F1 p=0.021 / AUC p=0.008（accuracy p=0.101 不显著，别说"全部显著"）；"research question three" 是新版编号（旧 RQ4 已改称 RQ3）；最后一句左右手分别代表两个数据集】

---

## 第 4 段 · C（3:00）

> Thanks, hzy. Two very different datasets — but we insisted on **one
> shared recipe**, so that every number is comparable.
>
> The recipe: **four models** — a decision tree, a random forest, logistic
> regression, and an SVM — crossed with **four imbalance strategies** —
> plain, SMOTE, cost-sensitive, and threshold moving — on **two datasets**,
> each under **two protocols**: 10-fold cross-validation plus a locked
> 70-30 holdout. That is **32 configurations**, one fixed seed, and every
> metric computed by a single shared evaluation script.
>
> Now the design rule I want you to remember: **nothing leaks**. All
> preprocessing — imputation, encoding, scaling — and SMOTE itself live
> **inside the pipeline**. So on every training fold, everything is
> fitted from scratch, and the validation fold is only transformed,
> never fitted. The threshold is chosen on inner folds and **locked
> before testing**. And we share one locked set of splits across the
> whole team — same rows, same folds, same seed. As a safety check, an
> equal-cost setting returns exactly the plain baseline — if those two
> ever disagree, something is broken.
>
> Let me show you it running live. [DEMO：Jupyter 里加载锁定切分 → 组装一条流水线 → 跑一个折 → 打印指标] One configuration, a few
> seconds, and out come the same metrics you will see in the results —
> fully reproducible. If the live demo fails, we have a recorded backup
> ready.
>
> That's the machine. **Now cyn will show you what it earned.**

【讲法：32 配方那句扳手指数；"nothing leaks" 是本段主题句，抬头说；demo 提前热机，界面字号调大；交棒句干脆】

---

## 第 5 段 · D（3:00）

> Thanks, cyw. So — what did all this earn us? Three findings.
>
> **Finding one: accuracy lies on imbalanced data.** Look at ZeroR on
> sick — the model that always says "negative". Its accuracy is **93.9%**
> — sounds great — and it catches **zero** patients. Zero. That's why
> every result we show is in F1, recall, and AUC, never accuracy alone.
>
> **Finding two: the winner flips across datasets.** On credit-g, the
> best configuration is **logistic regression plus the official 5-to-1
> cost matrix** — F1 0.597. On sick, it's **random forest plus SMOTE** —
> F1 0.897 — and from our exploratory ensemble study, **AdaBoost plus
> SMOTE pushes it to 0.937**. Why the flip? Not one reason — **four
> together**: sample size, signal strength, missingness, and imbalance.
> Here's the honest part: imbalance alone predicts the **wrong
> direction**. The small, clean dataset rewards the simple stable model;
> the big, noisy one rewards flexible ensembles. This is the
> cross-dataset comparison table — every claim has a significance test
> behind it.
>
> **Finding three: strategies buy recall — at a price that flips.** The
> recall gains are consistent: **8 of our 12 pre-specified comparisons
> survive Holm correction**. But whether F1 improves depends on the
> dataset. And one strong result from the ensemble study: on sick,
> AdaBoost beats decision tree plus SMOTE by **0.17 in F1, p < 0.001** —
> re-weighting beats re-sampling there. On credit-g it does not replicate
> — one more flip for the table.
>
> To wrap this up — **back to zyj.**

【讲法：三个 finding 各占一页，报标题时停一拍；"zero" 手势画零；对比表页停留最久，指着翻转的两行讲；0.937 与 p<0.001 放慢】

---

## 第 6 段 · E 收尾（1:30）

> Thanks, cyn. Three things to take home.
>
> **One**: on imbalanced data, choose your **metric** before you choose
> your model — accuracy will flatter you and lie to you. **Two**: there
> is no best model in general — only the model that matches the data's
> size, signal, and noise; our winners flipped for exactly that reason.
> **Three**: re-sampling and re-weighting are two different currencies —
> both can buy recall, but the exchange rate depends on the dataset.
>
> We are honest about limits: one fixed seed, and the ensemble study is
> exploratory — reported separately from the frozen matrix. Next steps:
> categorical-aware SMOTE, and decision thresholds tuned by expected
> cost, not accuracy.
>
> That's 〔组名〕 — thanks for listening. **We're happy to take your
> questions.**

【讲法：三句话逐条抬头，语速放慢；最后一句微笑环视，Q&A 开始】

---

## 衔接词速查表（锚点 · 勿单独改动）

| 交棒点 | 结尾交棒句（上一人） | 开头接榫句（下一人） |
|---|---|---|
| E→B | "Let's start with the messier one — Ling will take you into sick." | "Thanks, zyj. So — sick, from the UCI Thyroid Disease collection: 3,772 patient records, 30 attributes." |
| B→A | "…Now hzy will show you credit-g, and a clean experiment on why encoding matters." | "Thanks, Ling. After all those missing values, let me show you something much cleaner…" |
| A→C | "…one messy, one clean. C will show you how we model both with one shared pipeline." | "Thanks, hzy. Two very different datasets — but we insisted on one shared recipe…" |
| C→D | "That's the machine. Now cyn will show you what it earned." | "Thanks, cyw. So — what did all this earn us? Three findings." |
| D→E | "To wrap this up — back to zyj." | "Thanks, cyn. Three things to take home." |

## 排练与风险检查点
- **11-06 彩排①**：按本稿新顺序完整走一遍；A 是重点观察对象——卡壳即启动预案（A6 并给 D 一句话带过，A 保底讲满 2′ 数据介绍）
- **11-08 彩排②**：全真计时 15′；超支弹性位=D 段（先砍 Ensemble 表讲解，保对比表主叙事）；E 收尾固定 1′30″ 不膨胀
- demo 双保险：现场 Jupyter + 录屏备份（C 准备）；Q&A 按共享题库（每题库含 3 道 bagging/boosting 题）
- 稿件版本：B 段与 docs/B10_讲稿与幻灯要点_v1.1.md 同源；A/C/D/E 段为本次新拟初稿，**各自认领后可改内容、留衔接**
