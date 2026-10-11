"""A2+A3｜credit-g 字段字典与质量统计（角色 A 交付物）

产出：
- results/credit-g_schema.xlsx（sheet1 字段字典 / sheet2 数值五数概括）
- results/stats/credit-g_stats.md（质量统计，供 D 核对指标口径）
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "raw" / "credit-g.csv"
OUT_XLSX = ROOT / "results" / "credit-g_schema.xlsx"
OUT_STATS = ROOT / "results" / "stats" / "credit-g_stats.md"

NUMERIC_COLS = [
    "duration", "credit_amount", "installment_commitment",
    "residence_since", "age", "existing_credits", "num_dependents",
]

BIZ = {
    "checking_status": "现有支票账户状态（余额档位/无账户），信用最重要特征之一",
    "duration": "贷款期限（月）",
    "credit_history": "历史信用记录（按时还清/曾有拖欠/临界无记录等）",
    "purpose": "贷款用途（买车/家电/生意/教育等）",
    "credit_amount": "贷款金额（德国马克 DM）",
    "savings_status": "储蓄账户状态（余额档位/无已知储蓄）",
    "employment": "当前受雇年限档位（<1 至 ≥7 年/失业）",
    "installment_commitment": "分期还款占可支配收入比例（1–4 档）",
    "personal_status": "性别与婚姻状况组合",
    "other_parties": "共同债务人/担保人（无/担保人/共同申请人）",
    "residence_since": "现居住年限（1–4 档）",
    "property_magnitude": "名下财产（不动产/人寿保险/汽车/无）",
    "age": "申请人年龄（岁）",
    "other_payment_plans": "其他分期付款渠道（银行/商店/无）",
    "housing": "住房（自有/租房/免费提供）",
    "existing_credits": "在本行的现有信用笔数（1–4）",
    "job": "职业技能等级（失业/非技能/技能/高技能或管理层）",
    "num_dependents": "抚养人数（1–2）",
    "own_telephone": "是否有电话（none/yes）",
    "foreign_worker": "是否外国工人（yes/no）",
    "class": "目标变量：good（按时还款）/ bad（违约，正类）",
}


def build_schema(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        if col in NUMERIC_COLS:
            coltype = "数值"
            values = f"{s.min()} ~ {s.max()}"
        else:
            coltype = "名义（含 class）" if col == "class" else "名义"
            values = "；".join(f"{k}×{v}" for k, v in s.value_counts().items())
        rows.append({
            "列名": col,
            "类型": coltype,
            "业务含义": BIZ.get(col, ""),
            "缺失数": int(s.isna().sum()),
            "唯一值数": int(s.nunique()),
            "取值/分布": values,
        })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(CSV)
    schema = build_schema(df)
    five = df[NUMERIC_COLS].describe().T

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX) as w:
        schema.to_excel(w, sheet_name="字段字典", index=False)
        five.to_excel(w, sheet_name="数值五数概括")

    n_nominal = (schema["类型"] != "数值").sum()
    n_numeric = (schema["类型"] == "数值").sum()
    stats = f"""# credit-g 质量统计（A3，供 D 核对）

生成方式：`src/a2a3_creditg_schema_stats.py`，数据 `data/raw/credit-g.csv`（SHA-256 见 data/metadata/credit-g_source.md）

| 检查项 | 实际值 | 预期 | 结论 |
|---|---|---|---|
| 行 × 列 | {df.shape[0]} × {df.shape[1]} | 1000 × 21 | {'通过' if df.shape == (1000, 21) else '不一致'} |
| 类别分布 | {df['class'].value_counts().to_dict()} | good 700 : bad 300 | {'通过' if df['class'].value_counts().to_dict() == {'good': 700, 'bad': 300} else '不一致'} |
| 总缺失数 | {int(df.isna().sum().sum())} | 0 | {'通过' if df.isna().sum().sum() == 0 else '不一致'} |
| 重复行数 | {int(df.duplicated().sum())} | 0 | {'通过' if df.duplicated().sum() == 0 else '需处理'} |
| 名义列数（含 class） | {n_nominal} | 14 | {'通过' if n_nominal == 14 else '不一致'} |
| 数值列数 | {n_numeric} | 7 | {'通过' if n_numeric == 7 else '不一致'} |

## 数值属性五数概括（describe）

| 列 | min | Q1 | 中位数 | Q3 | max | 均值 |
|---|---|---|---|---|---|---|
"""
    for col, r in five.iterrows():
        stats += f"| {col} | {r['min']:.0f} | {r['25%']:.1f} | {r['50%']:.1f} | {r['75%']:.1f} | {r['max']:.0f} | {r['mean']:.1f} |\n"

    OUT_STATS.parent.mkdir(parents=True, exist_ok=True)
    OUT_STATS.write_text(stats, encoding="utf-8")
    print(f"已写入 {OUT_XLSX}")
    print(f"已写入 {OUT_STATS}")
    print(f"名义 {n_nominal} 列 / 数值 {n_numeric} 列；缺失 {int(df.isna().sum().sum())}；重复行 {int(df.duplicated().sum())}")


if __name__ == "__main__":
    main()
