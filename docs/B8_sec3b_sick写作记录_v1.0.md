# B8 写作记录 v1.0（2026-10-27）

## 交付物
- `docs/B8_sec3b_sick_draft_v1.0.tex` —— sec3b_sick 完整英文成稿（约 1 页 + 3 图）

## 存档说明
本稿存于 docs/，与 D 的 sec6/sec7 草稿同批处理，经核验后写入 Overleaf：
```bash
tail -n +6 docs/B8_sec3b_sick_draft_v1.0.tex > overleaf/sections/sec3b_sick.tex
```

## 数字来源
| 稿内数字 | 来源 |
|---|---|
| 3,772×30；3,541:231；61 重复 | results/stats/sick_missing_audit.csv + B2 复核表 |
| 6,064 (5.4%)；TBG 100%；配对零异常；1,129 行 (29.9%) | B3 审计（results/stats/） |
| B4 三方案 0.830±0.049 / 0.843±0.069 / AUC 0.974 vs 0.964；n=2,643；-19 正类 | results/b4_missing_comparison.csv（经 D_B4指标口径复核） |
| 精度保持 0.94 对 0.91、0.95 对 0.60 | 主矩阵 summary（D4 已核验） |
| UCI 102 / DOI / Garavan 1986 / CC BY 4.0 / 7,200 条不适用 | data/metadata/sick_source.md（哈希已更新） |

## 结构
正文 3 段（来源与标签语义 → 缺失审计机制 → 处理决策+B4 论证+防泄漏）+ 结尾 RF 精度保持伏笔句（指向 sec:results 的 Ensemble Study）+ 图 1/2 并排（类别分布 + 缺失率）+ 图 3（箱线图，含"单指标不可分"解读）。

## 已知依赖
- `\ref{sec:results}` 依赖 D 的 sec6 label（D 草稿已定义）；若 E 统稿改名需同步。
- 三图 overleaf/figures/fig_b_*.pdf 均已在库。
- 图注含数据集、协议、模型、指标、正类信息，符合组内 figure-rules。
