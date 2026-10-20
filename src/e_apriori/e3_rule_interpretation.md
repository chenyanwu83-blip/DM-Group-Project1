# E3 credit-g Apriori：规则表解读

## 数据与方法

输入是 E2 已确认的 `credit_g_transactions.csv`（1,000 条记录、85 个布尔项，`class=bad` 为 300 条；SHA-256：`8a067ddcba7c3885aba2883368ff69d1b04a27af83a473b67a510ab23fb38747`）。只用于关联规则挖掘，不作为监督模型的特征输入。

使用 `mlxtend` 的 `apriori(..., use_colnames=True, max_len=3)`，将最低支持度从 0.30、0.25、0.20、0.15、0.10 扫到 0.05。对应频繁项集数为 292、431、690、1,358、2,984、7,732；完整扫描见 `e3_support_scan.csv`。在 0.05 下生成 41,052 条方向性规则，其中后件**恰好**为 `class=bad` 的有 310 条。进一步要求 confidence ≥ 0.50，剩 14 条；在 0.10 支持度时符合该置信度门槛的只有 4 条，因此以 0.05 作探索门槛。

为减少重复表述，比较双条件规则的 confidence 与两个单条件各自的 bad 比例，取较高者为基准；差值至少 5 个百分点的 6 条作为代表候选。这个差值是描述性筛选标准，不能当作因果增益或统计显著性。

## 六条代表规则

所有规则的后件均为 `class=bad`。`support` 是同时符合左侧条件且为 bad 的人数除以 1,000；`confidence` 是符合左侧条件的人中 bad 的比例；`lift` 是该比例除以全体 bad 比例 0.30。精确数值见 `e3_selected_rules.csv`。

| 左侧条件 | 符合条件 | 其中 bad | support | confidence | lift | 解读 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `duration=q5_of_5` 且 `savings_status=<100` | 124 | 71 | 7.1% | 57.3% | 1.91 | 最长期限档与低储蓄类别共同出现时，样本中的 bad 比例较高。 |
| `credit_amount=q5_of_5` 且 `savings_status=<100` | 116 | 61 | 6.1% | 52.6% | 1.75 | 最高金额档与低储蓄类别共同出现时，bad 比例高于总体。 |
| `checking_status=<0` 且 `installment_commitment=4` | 142 | 80 | 8.0% | 56.3% | 1.88 | 账户状态类别与原始第 4 档还款负担共同出现时，bad 比例较高。 |
| `checking_status=<0` 且 `job=skilled` | 172 | 96 | 9.6% | 55.8% | 1.86 | 两个类别的组合与 bad 相关；`job=skilled` 单独出现时 bad 比例约 29.5%，不能把组合结果归因于工作类别。 |
| `duration=q5_of_5` 且 `credit_history=existing paid` | 97 | 50 | 5.0% | 51.5% | 1.72 | 期限档与原始信用历史类别共同出现时，bad 比例较高；`existing paid` 是数据集原标签，不应改写为“逾期”。 |
| `duration=q5_of_5` 且 `installment_commitment=4` | 104 | 53 | 5.3% | 51.0% | 1.70 | 最长期限档与第 4 档还款负担共同出现时，bad 比例高于总体 30% 基准。 |

六条规则的条件人数与 bad 人数已由脚本直接从交易表重新计数，均与规则输出一致。`duration=q5_of_5` 在本数据中对应 30–72，`credit_amount=q5_of_5` 对应 4,736–18,424；区间由 E2 的等频分箱得到，不能推广到别的数据集。

## 解释边界

这些是同一份 1,000 条数据上扫描并筛选出的**探索性关联**，没有独立样本验证稳定性。尤其第 5、6 条的 support 接近 5% 门槛，可能受样本与分箱边界影响。confidence 不是模型预测准确率；lift 大于 1 也不能证明左侧条件导致 bad。报告和展示中应同时给出人数、support、confidence 与 lift，并使用“相关”“共同出现”等表述。
