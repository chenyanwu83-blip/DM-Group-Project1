# D_report_20261102 · sec6 + 7.1 交付（tex / Word）

生成时间：2026-11-02 晚。生成前已与本人确认基线。

## 版本基线（已确认）

- **文字**：`D_Report_Final_20261102.docx`（GPT Final 版）逐字转写，
  叠加三处修正：
  1. 删除 "6.1 Main Results continued" 临时排版标题；
  2. 6.3 标题 "Cross Dataset" → "Cross-Dataset"；
  3. 6.4 标题 "Trade offs" → "Trade-offs"。
- **图表**：Final 内嵌的 4 张最新渲染图（Mean F1 条形图、ROC/PR、策略增益、
  轴向已修正的混淆矩阵），已提取为本目录 PNG。
- **数字**：此前已对 `results/*.csv` 台账全量核验（零矛盾）；本轮新增复算
  sick LR SMOTE 的 recall 0.606→0.870 / precision 0.803→0.373，与 F1 0.521 吻合。
- 说明：docs/ 下 D7_tex v2.2 的三处修正（ZeroR 措辞 / 21 categorical /
  分通道 SMOTE）针对的是**旧长稿线**的句子，GPT Final 重写时已不含这些表述，
  故不适用本交付。

## 文件清单

| 文件 | 说明 |
|---|---|
| `D_Report_sec6-7_20261102.tex` | LaTeX 源（12pt article，A4，booktabs 表，pdfLaTeX/XeLaTeX 均可编译，Overleaf 直接可用） |
| `D_Report_sec6-7_20261102.pdf` | 本地编译版（tectonic 0.17.0，5 页 A4，零错误） |
| `D_Report_sec6-7_20261102.docx` | Word 版（pandoc 3.8 转换：8 标题带编号、5 表、4 图内嵌） |
| `fig_6_1_main_f1.png` | Figure 6.1 主矩阵 Mean F1 条形图 |
| `fig_6_2_roc_pr.png` | Figure 6.2 RF ROC/PR 曲线 |
| `fig_6_3_strategy_gain.png` | Figure 6.3 策略增益图 |
| `fig_6_4_confusion.png` | Figure 6.4 混淆矩阵（行=实际，列=预测） |

## 修订记录

- **修订 1（初版）**：Final + 三处修正（删 continued 标题、两处连字符），tex/docx/PDF 三格式。
- **修订 2（按模拟评阅四项）**：
  - [a] 表内数值整值改单一数学对象 + docx 固定列宽（tblLayout fixed + gridCol），修复 Word 断行断词；
  - [b] 6.2 表注补 "Following the corrected paired-test procedure of Section 5" 指向；
  - [c] 6.3-1 重写为发现先行（86.3% / 44.1% / 88.7% / 0.6% 负类误报）；6.5-1 用 Bagging+SMOTE（0.987→F1 0.785）vs AdaBoost+SMOTE（0.931/0.937）对比替换泛泛开头；
  - [d] 7.1 RQ3/RQ4 标题改结果句，0.570 明确标注 holdout。
- **修订 3（发现并修复 pandoc 丢段 bug）**：pandoc 会整段丢弃 `\noindent{\small ...}` 表注
  （统计口径、±SD 说明、5:1/equal-cost 说明、表来源此前在 Word 版全部缺失——模拟评阅的第①③条
  对 Word 版实际成立）。已改为普通段落，Word 版现在与 tex/PDF 内容一致。
  sick 代价臂注释同步改为 "equal-cost 1:1 control" 显式化实验设计。
- 当前状态：tex/PDF 6 页，docx 渲染 6 页；断行/断词检查清零；正文 12pt（docDefaults 确认）。

- **修订 4（第二轮模拟评阅，87→90 路径）**：
  - [f] 表宽修复：`tabcolsep` 6→4pt、表体 footnotesize、重算列宽——原 p{} 列宽漏算每列 2×6pt 填充，
    表 6.1/6.2/6.4/6.5 总宽 6.9–7.3in 超出 6.27in 正文宽度（横线顶到页缘，评阅属实）；
  - [g] 每张图/表连同图注/表题包进 minipage（原子不可拆页），图注断页与近空尾页消除；
  - [h] 6.3-1："prices both"→"evaluates the decision trade-offs of both configurations"（sick 无官方代价矩阵），
    正文只留百分比、计数指向图 6.4；
  - [i] 6.5-1 末句改为 "The precision–recall trade-off depends on the model, even under the same resampling strategy."；
  - [j] Table 6.5 补 credit-g 最高均值加粗（Bagging+SMOTE 0.548，与表注一致）；
  - [k] 7.1 RQ3 "best AUC" 限定为 "among the three tested missing-data schemes"；
  - 附：`\emergencystretch` + 表 6.4 注释句序调整，修复 texttt 长文件名造成的 36pt 溢出。
  - 验证：PDF 5 页 A4，逐页 bbox 检查右边界零溢出（阈值 524pt），尾页充实；
    Word 版断行清零、内容抽查全过。

- **修订 5（合并 B 交付的脚注要素，闭环评审 D5 + 批注 #12/#13）**：6.3 第 3 条改为
  A6 同款"三方案对照式"——并入 ±SD（0.830±0.049 / 0.843±0.069）、"19 of the 231 positives"
  分母、决策句 "Decision: impute inside the pipeline" 与 `b4_missing_comparison.csv` 出处
  （来源：`docs/B8_sec6_footnote_for_D_v1.0.tex`，git ce7ffa2/b5d9763）；保留本稿独有的
  marker 臂 AUC 0.965。B 的修订记录可标 ✅。验证：PDF 5 页零溢出，docx 断行 0。

## 与其他版本的关系

- `03_报告修改_截止1104/D_Report_sec6-7_v2.3_20261102.docx/.pdf`：同一内容基线的
  docx/PDF 渲染（v2.3 = Final + 同样三处修正）。
- `DM-Group-Project1/docs/D7_sec6_results_draft_v2.2.tex`：另一条长稿线
  （6.1–6.7 老结构），含 v2.2 三处修正，供 E 整合 overleaf 时参考取舍。
- 待组长确认后仍可能变动：批注 27（6.1/6.2 合并）、批注 29（策略增益图删否）。
