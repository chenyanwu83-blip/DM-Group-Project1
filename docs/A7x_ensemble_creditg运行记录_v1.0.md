# A 侧 ensemble study 镜像运行记录 v1.0 — credit-g（2026-10-22）

> 镜像 B 的 b7x_ensemble_study（V4 §1.3 集成增强研究），A 执行 credit-g 侧：
> `python src/b7x_ensemble_study.py --dataset creditg --all`（cds535 环境，实跑约 10 秒）。
> 入口/切分/口径与 sick 侧完全同源：D 锁定切分 + evaluate.py 统一指标 + 模板零偏差列 +
> 独立命名空间 results/ensemble/，不碰主矩阵冻结规格。落盘自查（列/折数/sha/preds 4 配置×2 协议）全 PASS。

## 结果（cv10 均值，正类 bad；holdout 见各 *_holdout_summary.csv）

| 配置 | acc | recall | F1 | AUC | cost | holdout F1 |
|---|---|---|---|---|---|---|
| bag_plain | 0.756 | 0.430 | 0.510 | 0.782 | 0.928 | 0.448 |
| bag_smote | 0.723 | 0.560 | **0.548** | 0.770 | 0.805 | 0.543 |
| ada_plain | 0.717 | 0.473 | 0.499 | 0.735 | 0.915 | 0.436 |
| ada_smote | 0.717 | 0.483 | 0.501 | 0.723 | 0.903 | 0.500 |

主矩阵参照（cv10）：lr_cost F1 0.597 / rf_cost 0.595 / lr_smote 0.591；总代价最优 lr_cost 0.514。

## 关键发现：ada 优势不复现，集成增益是数据集依赖的（ch6 核心素材）

1. **赢家翻转**：sick 上 ada_smote（F1 0.937）登顶并超主矩阵全部 16 配置；credit-g 上集成最优
   bag_smote 仅 0.548，ada_smote 0.501，双双低于主矩阵冠军（lr_cost 0.597，差 5–10 个点）。
   holdout 同方向（bag_smote 0.543 / ada_smote 0.500 < lr_smote 0.587）。
2. **排序也翻转**：sick 上 ada ≫ bag；credit-g 上 bag_smote ＞ ada_smote——「重加权」在稀疏名义、
   近可分、94:6 的 sick 有效；「聚合+重采样」在稠密混合属性、70:30 的 credit-g 略优。
3. **精度不崩但不赚**：credit-g 上 bag/ada 精度保持（0.72–0.76），与 sick 结论一致的一半；
   但无代价敏感通道，总代价 0.80–0.93 全部劣于 lr_cost 0.514——credit-g 的最优解仍是
   「线性模型 + 官方 5:1 代价矩阵」，集成无法替代代价信息。
4. **写作建议**（供 D6/E6）：与 A6（编码消融）串联成 RQ4 的完整故事——
   「方法族 × 数据集特性」匹配：credit-g 赢在代价敏感线性模型，sick 赢在重采样集成，
   无单一方法双料登顶。

## 移交

- D：可并入 D4/D6 跨数据集对比讨论（ensemble 为扩展研究，不计入 32 配置主矩阵与 t 检验）。
- E：ch6 无监督小节后的「扩展研究」段或 ch7 future work 可引（AdaBoost/Bagging 未进课程 4 算法清单，作增强讨论）。
