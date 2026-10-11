# credit-g 质量统计（A3，供 D 核对）

生成方式：`src/a2a3_creditg_schema_stats.py`，数据 `data/raw/credit-g.csv`（SHA-256 见 data/metadata/credit-g_source.md）

| 检查项 | 实际值 | 预期 | 结论 |
|---|---|---|---|
| 行 × 列 | 1000 × 21 | 1000 × 21 | 通过 |
| 类别分布 | {'good': 700, 'bad': 300} | good 700 : bad 300 | 通过 |
| 总缺失数 | 0 | 0 | 通过 |
| 重复行数 | 0 | 0 | 通过 |
| 名义列数（含 class） | 14 | 14 | 通过 |
| 数值列数 | 7 | 7 | 通过 |

## 数值属性五数概括（describe）

| 列 | min | Q1 | 中位数 | Q3 | max | 均值 |
|---|---|---|---|---|---|---|
| duration | 4 | 12.0 | 18.0 | 24.0 | 72 | 20.9 |
| credit_amount | 250 | 1365.5 | 2319.5 | 3972.2 | 18424 | 3271.3 |
| installment_commitment | 1 | 2.0 | 3.0 | 4.0 | 4 | 3.0 |
| residence_since | 1 | 2.0 | 3.0 | 4.0 | 4 | 2.8 |
| age | 19 | 27.0 | 33.0 | 42.0 | 75 | 35.5 |
| existing_credits | 1 | 1.0 | 1.0 | 2.0 | 4 | 1.4 |
| num_dependents | 1 | 1.0 | 1.0 | 1.0 | 2 | 1.2 |
