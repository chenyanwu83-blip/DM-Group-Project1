# E4 credit-g：Apriori 与决策树、随机森林的可解释性对照

## 比较口径

本表使用 E3 的六条探索性规则（`e3_selected_rules.csv`），以及 C3 的 `creditg_dt_plain_v1_run1`、`creditg_rf_plain_v1_run1`。两种树模型均按 C 的固定配置、原始 20 个特征和固定 700 条训练记录重建；300 条留出记录的**预测类别与 bad 概率逐条复现** C3 已保存的结果。数据文件 SHA-256：`d0baf9fddd41e5a6af0ba84e6037a415f11384dd40c494186563a2d3b68f5c25`。

E3 Apriori 在全部 1,000 条记录上探索关联，三项连续字段另按 A5 分为五档；C3 的树模型仅用训练集拟合，连续字段保留原值。因此下面是**解释视角的定性对照**，不是独立验证，也不能把 Apriori confidence 与树模型的预测准确率直接比较。所有 E3 规则的右侧均为 `class=bad`。

## 六条规则并排对照

| E3 条件（support；confidence；lift） | 决策树路径中有无对应字段 | RF 字段重要性（20 个字段中的名次） | 对照结论 |
| --- | --- | --- | --- |
| `duration=q5_of_5` + `savings_status=<100`（7.1%；57.3%；1.91） | 使用 `duration` 原值阈值；**未使用** `savings_status` | `duration` 第 2（0.1074）；`savings_status` 第 9（0.0417） | 仅“期限”字段呼应；未重现完整两条件规则。 |
| `credit_amount=q5_of_5` + `savings_status=<100`（6.1%；52.6%；1.75） | 使用 `credit_amount` 原值阈值；**未使用** `savings_status` | `credit_amount` 第 1（0.1409）；`savings_status` 第 9（0.0417） | 仅“金额”字段呼应；未重现完整两条件规则。 |
| `checking_status=<0` + `installment_commitment=4`（8.0%；56.3%；1.88） | 根节点使用 `checking_status`，但分组包含 `<0` **和** `0<=X<200`；未使用还款负担档位 | `checking_status` 第 3（0.1066）；`installment_commitment` 第 11（0.0401） | 账户状态在三种视角都出现；树路径并未单独确认 `<0` 加第 4 档的组合。 |
| `checking_status=<0` + `job=skilled`（9.6%；55.8%；1.86） | 使用 `checking_status`；**未使用** `job` | `checking_status` 第 3（0.1066）；`job` 第 15（0.0274） | 只在账户状态字段上部分呼应；不能将组合关联归因于职业。 |
| `credit_history=existing paid` + `duration=q5_of_5`（5.0%；51.5%；1.72） | 使用 `duration`；**未使用** `credit_history` | `duration` 第 2（0.1074）；`credit_history` 第 6（0.0507） | 仅期限字段呼应；`existing paid` 是原始类别名，不代表“逾期”。 |
| `duration=q5_of_5` + `installment_commitment=4`（5.3%；51.0%；1.70） | 使用 `duration`；**未使用** `installment_commitment` | `duration` 第 2（0.1074）；`installment_commitment` 第 11（0.0401） | 仅期限字段呼应；未重现完整两条件规则。 |

RF 数值来自 `feature_importances_`，表示该模型在训练时各字段分裂带来的不纯度下降份额，总和为 1；它**不表示**字段取值越高越容易 bad，也不表示因果影响。连续字段或取值较多的字段可能因可选切点较多而获得较高分数。完整 20 项见 `e4_rf_feature_importance.csv`。

## 一条可核查的决策树路径

导出的决策树深度为 6，有 13 个叶子，根节点先按 `checking_status` 分裂。在训练集中，一条预测为 bad 的路径是：`checking_status` 属于 `<0` **或** `0<=X<200` → 原始 `duration > 31.5` → `employment` 不属于 `unemployed`；该叶有 **51 bad、23 good（74 条训练记录）**。这说明树按顺序对一组记录做分类，不能表述为“`checking_status=<0` 单独造成坏信用”。E3 的 `duration=q5_of_5` 是分箱标签，包含 30–72；树的 `>31.5` 是另一阈值，二者并不完全相同。完整路径和类别编码对照见 `e4_decision_tree_paths.txt`。

树通道沿用 C2 的 `OrdinalEncoder`。类别编码数字只是字母排序后的代码，例如根节点 `checking_status <= 1.5` 对应 `{0<=X<200, <0}`；**不能**把它当作余额大小有序的数值。E4 没有改动 C 的预处理或模型参数。

## 可用于报告的素材段落

在 credit-g 上，Apriori 的六条探索性规则反复涉及贷款期限、金额及账户状态；C3 plain 随机森林也将金额、期限和账户状态列为前三项重要字段。决策树首先按账户状态分裂，并在部分分支使用期限和金额。三种方法在**字段层面**有交集，但该树没有使用六条规则中的储蓄状态、还款负担、职业和信用历史字段，因此没有一条双条件规则被树路径完整重现。Apriori 描述全样本中条件与 bad 标签的共同出现，树路径描述模型对训练记录的顺序判断，随机森林重要性概括模型整体使用字段的程度；这些证据不可相互替代，更不能推出因果关系。

## 复现与限制

在仓库根目录运行 `PYTHONPATH=. python src/e_apriori/e4_extract_tree_rf.py`。脚本读取固定索引与 C3 已保存的留出预测，逐条核对两模型的类别和概率，再导出 `sklearn.tree.export_text` 与 `RandomForestClassifier.feature_importances_`。C3 的 `plain` 策略不含重采样，因此此处用标准 `sklearn.pipeline.Pipeline` 重建同一预处理和分类器；**300/300 逐条一致**是等价性检查。E3 的六条规则在同一 1,000 条样本中经多次筛选选出，存在挑选偏差；树的重要性和路径也随训练样本、编码和模型参数变化。报告中应称“部分呼应”“字段层面一致”，不要写成外部验证或因果结论。
