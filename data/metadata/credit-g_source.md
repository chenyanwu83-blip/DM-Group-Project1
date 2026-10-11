# credit-g 数据来源记录

| 项目 | 内容 |
|---|---|
| 数据集 | Statlog (German Credit Data) |
| UCI ID | 144 |
| DOI | 10.24432/C5NC77 |
| 捐赠者 | Hans Hofmann（汉堡大学），Statlog Project, 1994 |
| 许可 | CC BY 4.0 |
| 获取日期 | 2026-10-11 |
| 获取方式 | 下载 OpenML credit-g 变体 ARFF（data/raw/credit-g.arff，语义列名 + good/bad 标签），经 load_data.py 解码转存 CSV；与 UCI 官方原始版（Attribute1-20 编码 + 1/2 标签）为同一数据 |
| 本地文件 | data/raw/credit-g.csv |
| SHA-256 | `d0baf9fddd41e5a6af0ba84e6037a415f11384dd40c494186563a2d3b68f5c25` |

## 完整性核验

- 行 × 列：1000 × 21（预期 1000 × 21）
- 类别分布：{'good': 700, 'bad': 300}（预期 good 700 : bad 300）
- 与 ucimlrepo 官方副本交叉核验：shape 一致=通过；类别分布一致(1=good,2=bad)=通过；列数一致(命名变体除外)=通过；列名为 OpenML 语义命名（UCI 官方为 Attribute1-20 编码），属同一数据的两个变体

## 引用格式（BibTeX，供 E 入引用库）

```bibtex
@misc{uci_creditg,
  author       = {Hofmann, Hans},
  title        = {{Statlog (German Credit Data)}},
  year         = {1994},
  howpublished = {UCI Machine Learning Repository},
  note         = {ID 144, DOI 10.24432/C5NC77, License CC BY 4.0}
}
```
