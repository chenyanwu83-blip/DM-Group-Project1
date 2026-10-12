# credit-g 图注（A4 交付，数字由 src/a4_creditg_eda.py 自动生成）

## fig1_class_distribution.png
- **Caption (EN)**: Class distribution of the credit-g dataset: 700 good vs. 300 bad (30% positive rate), a 70:30 class imbalance.
- **解读**：正类 bad 占 30%，属中度不平衡；多数类基线 accuracy=70%，是 RQ2「accuracy 何时失效」的下限参照。

## fig2_checking_status_class.png
- **Caption (EN)**: Stacked distribution of class by checking_status, sorted by bad rate. Overdrawn accounts ("<0") show the highest default rate (49.3%), while "no checking" the lowest (11.7%).
- **解读**：支票账户状态与违约呈清晰梯度，与"账户余额越少、信用风险越高"的信贷直觉一致：

| checking_status | 样本数 | bad | bad 率 |
|---|---|---|---|
| <0 | 274 | 135 | 49.3% |
| 0<=X<200 | 269 | 105 | 39.0% |
| >=200 | 63 | 14 | 22.2% |
| no checking | 394 | 46 | 11.7% |

  bad 率从 49.3%（<0）单调降至 11.7%（no checking），横跨整体 30% 基线两侧——验证 checking_status 为最强预测特征之一；报告 ch3 可引用该表。

## fig3_numeric_hist.png
- **Caption (EN)**: Histograms (bins=20) of duration, credit_amount and age. All three are right-skewed; credit_amount ranges from 250 to 18424 DM.
- **解读**：三个数值属性均右偏（中位数 2320 DM ≪ 最大 18424 DM），支持 A5 规格单在等频离散化（quantile binning）中使用分位数切分而非等宽，避免高频箱塌缩。
