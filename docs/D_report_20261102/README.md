# D_report_20261102 · sec6 + 7.1（tex / Word）

第 6 章（结果与评估）和 7.1 节（结论中 D 负责的部分）的交付，2026-11-02 定稿。
正文里的所有数字都和 `results/` 下的 CSV 核对过（含 sick LR SMOTE 的
recall 0.606→0.870 / precision 0.803→0.373 与 F1 0.521 的复算）。

## 文件清单

| 文件 | 说明 |
|---|---|
| `D_Report_sec6-7_20261102.tex` | LaTeX 源（12pt article，A4，booktabs 表，pdfLaTeX/XeLaTeX 均可编译，Overleaf 直接可用） |
| `fig_6_1_main_f1.png` | Figure 6.1 主矩阵 Mean F1 条形图 |
| `fig_6_2_roc_pr.png` | Figure 6.2 RF ROC/PR 曲线 |
| `fig_6_3_strategy_gain.png` | Figure 6.3 策略增益图 |
| `fig_6_4_confusion.png` | Figure 6.4 混淆矩阵（行=实际，列=预测） |

编译出的 PDF 是 5 页 A4；需要 Word 版的话用 pandoc 转一下就有（8 个带编号的标题、
5 张表、4 张图都会带上），pdf/docx 产物就没放进仓库。

## 改了几版

1. 初版：正文 + 三处排版修正（删 "continued" 临时标题、"Cross-Dataset"/"Trade-offs" 连字符）。
2. 表内 ±SD 改成一个数学对象、docx 固定列宽，修 Word 里的断行断词；6.2 表注补了对
   Section 5 检验流程的指向；6.3/6.5 开头改成先把发现摆出来；7.1 的 RQ3/RQ4 小标题
   改成结果句，0.570 标明是 holdout。
3. 发现 pandoc 转 docx 时会整段丢掉 `\noindent{\small ...}` 写的表注，四条表注在
   Word 版里全没了——改成普通段落后就一致了。
4. 表宽按 `tabcolsep=4pt` + `footnotesize` 重算（原来超正文宽度，横线顶到页缘）；
   每张图/表连同图注包进 minipage，不再被分页拆开。
5. 6.3 第 3 条按 B 的建议改成三方案对照（±SD、19 of 231 分母、
   "Decision: impute inside the pipeline"、数据来自 `b4_missing_comparison.csv`），
   保留了本稿独有的 marker 臂 AUC 0.965。

## 和其他版本的关系

- 另有同内容的 Word/PDF 版本（v2.3，随交付另行提交，不入库）。
- `docs/D7_sec6_results_draft_v2.1.tex` 起的 D7 草稿系列：更早的长稿（6.1–6.7 老结构），
  供 E 整合 overleaf 时参考取舍。
