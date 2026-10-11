"""A1｜credit-g 来源核验与哈希记录（角色 A 交付物）

产出 data/metadata/credit-g_source.md：
- 本地 data/raw/credit-g.csv 的 SHA-256（供 E 引用库、报告来源关卡）
- 用 ucimlrepo 官方副本交叉核验 shape / 列名 / 类别分布（网络失败不影响出报告）
"""

import hashlib
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "raw" / "credit-g.csv"
META_DIR = ROOT / "data" / "metadata"
OUT = META_DIR / "credit-g_source.md"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_cross_check(local: pd.DataFrame) -> tuple[bool, str]:
    """本地 CSV 是 OpenML credit-g 变体（语义列名 + good/bad）；
    ucimlrepo 官方版是 UCI 原始编码（Attribute1-20 + 1/2，1=good, 2=bad）。
    故按 shape + 语义对齐后的类别分布核验，列名差异属预期变体差异。"""
    try:
        from ucimlrepo import fetch_ucirepo
        off = fetch_ucirepo(id=144).data.original
        off_dist = off["class"].map({1: "good", 2: "bad"}).value_counts().to_dict()
        local_dist = local["class"].value_counts().to_dict()
        checks = {
            "shape 一致": off.shape == local.shape,
            "类别分布一致(1=good,2=bad)": off_dist == local_dist,
            "列数一致(命名变体除外)": off.shape[1] == local.shape[1],
        }
        ok = all(checks.values())
        detail = "；".join(f"{k}={'通过' if v else '不一致'}" for k, v in checks.items())
        return ok, detail + "；列名为 OpenML 语义命名（UCI 官方为 Attribute1-20 编码），属同一数据的两个变体"
    except Exception as e:  # 网络问题不阻塞交付
        return False, f"官方副本拉取失败（{type(e).__name__}: {e}），仅记录本地哈希"


def main() -> None:
    local = pd.read_csv(CSV)
    digest = sha256_of(CSV)
    cross_ok, cross_detail = official_cross_check(local)
    class_dist = local["class"].value_counts().to_dict()

    META_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""# credit-g 数据来源记录

| 项目 | 内容 |
|---|---|
| 数据集 | Statlog (German Credit Data) |
| UCI ID | 144 |
| DOI | 10.24432/C5NC77 |
| 捐赠者 | Hans Hofmann（汉堡大学），Statlog Project, 1994 |
| 许可 | CC BY 4.0 |
| 获取日期 | {date.today().isoformat()} |
| 获取方式 | 下载 OpenML credit-g 变体 ARFF（data/raw/credit-g.arff，语义列名 + good/bad 标签），经 load_data.py 解码转存 CSV；与 UCI 官方原始版（Attribute1-20 编码 + 1/2 标签）为同一数据 |
| 本地文件 | data/raw/credit-g.csv |
| SHA-256 | `{digest}` |

## 完整性核验

- 行 × 列：{local.shape[0]} × {local.shape[1]}（预期 1000 × 21）
- 类别分布：{class_dist}（预期 good 700 : bad 300）
- 与 ucimlrepo 官方副本交叉核验：{cross_detail}

## 引用格式（BibTeX，供 E 入引用库）

```bibtex
@misc{{uci_creditg,
  author       = {{Hofmann, Hans}},
  title        = {{{{Statlog (German Credit Data)}}}},
  year         = {{1994}},
  howpublished = {{UCI Machine Learning Repository}},
  note         = {{ID 144, DOI 10.24432/C5NC77, License CC BY 4.0}}
}}
```
""", encoding="utf-8")
    print(f"已写入 {OUT}")
    print(f"SHA-256: {digest}")
    print(f"交叉核验: {cross_detail}")


if __name__ == "__main__":
    main()
