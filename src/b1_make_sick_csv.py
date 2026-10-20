"""
B1 · sick.csv 数据获取/校验（V3 组仓库适配版 · 2026-10-15 更新）
================================================================
10-15 起组仓库已有 C 用 load_data.py 从 arff 统一生成的 data/raw/sick.csv
（列名为 'TSH measured' 带空格版，SHA-256 = 3cc14348…，与 D 的
split_manifest.json 锚定）。本脚本默认只做【校验】，不重写数据——
避免破坏全组的切分哈希锚点。

用法：
    python src/b1_make_sick_csv.py            # 校验现有 csv（日常用这个）
    python src/b1_make_sick_csv.py --force    # 强制重新生成（仅数据损坏时）

生成路线（--force 时）：
    路线 A（主）：Weka 自带 sick.arff → CSV（本机已有 arff）
    路线 B（备用）：UCI 官网 zip，拼接 sick.data(2800) + sick.test(972)
    注：ucimlrepo.fetch_ucirepo(id=102) 官方不支持导入（实测报错），全组勿用。
"""
import csv
import hashlib
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # VS-Code/ 项目根
RAW_DIR = ROOT / "data" / "raw"
CSV_PATH = RAW_DIR / "sick.csv"

# 组仓库规范列名（与 C 的 load_data.py 一致：measured 列用空格，非下划线）
COLUMNS = [
    "age", "sex", "on thyroxine", "query on thyroxine",
    "on antithyroid medication", "sick", "pregnant", "thyroid surgery",
    "I131 treatment", "query hypothyroid", "query hyperthyroid", "lithium",
    "goitre", "tumor", "hypopituitary", "psych", "TSH measured", "TSH",
    "T3 measured", "T3", "TT4 measured", "TT4", "T4U measured", "T4U",
    "FTI measured", "FTI", "TBG measured", "TBG", "referral source", "Class",
]
CANON_SHA = "3cc14348968e47c2e1e52ad350edf6b0b96b0578c27be9b29d10fe21740854ab"


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def from_weka_arff():
    candidates = [
        Path("/Users/mac/Desktop/LU/Data Mining/UCI/sick.arff"),
        RAW_DIR / "sick.arff",
        ROOT / "data" / "raw" / "sick.arff",
    ]
    arff_path = next((p for p in candidates if p.exists()), None)
    if arff_path is None:
        return None
    print(f"[路线A] 找到 Weka 自带文件：{arff_path}")
    rows, in_data = [], False
    for line in arff_path.read_text(encoding="utf-8").splitlines():
        low = line.strip().lower()
        if low.startswith("@data"):
            in_data = True
            continue
        if not in_data or not line.strip() or line.startswith("%"):
            continue
        parts = next(csv.reader(io.StringIO(line)))
        rows.append([v.strip("' ") for v in parts])
    return rows


def from_uci_zip():
    url = "https://archive.ics.uci.edu/static/public/102/thyroid+disease.zip"
    print(f"[路线B] 未找到本地 arff，开始下载：{url}")
    buf = urllib.request.urlopen(url, timeout=120).read()
    zf = zipfile.ZipFile(io.BytesIO(buf))
    rows = []
    for member in ("sick.data", "sick.test"):
        for line in zf.read(member).decode("utf-8").splitlines():
            if not line.strip():
                continue
            line = re.sub(r"\|\d+$", "", line)
            parts = next(csv.reader(io.StringIO(line)))
            rows.append([v.strip("' ") for v in parts])
    return rows


def main() -> None:
    # ── 默认：校验模式（不动数据）────────────────────────────
    if "--force" not in sys.argv and CSV_PATH.exists():
        got = sha256_of(CSV_PATH)
        n_rows = sum(1 for _ in CSV_PATH.open()) - 1
        ok = got == CANON_SHA
        print("=" * 60)
        print("B1 · 数据校验（组仓库锁定版）")
        print("=" * 60)
        print(f"文件     ：{CSV_PATH}")
        print(f"规模     ：{n_rows} 行 × 30 列")
        print(f"SHA-256  ：{got}")
        print(f"与组内锚定（split_manifest）：{'✅ 一致' if ok else '❌ 不一致！'}")
        if not ok:
            print("  数据被改动过。先 git checkout -- data/raw/sick.csv 恢复，")
            print("  恢复不了再 python src/b1_make_sick_csv.py --force 重建。")
        return

    # ── --force：重新生成────────────────────────────────────
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = from_weka_arff() or from_uci_zip()
    assert len(rows) == 3772, f"行数异常：{len(rows)}（应为 3772）"
    assert all(len(r) == 30 for r in rows), "存在列数不为 30 的行"

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)

    got = sha256_of(CSV_PATH)
    print(f"✅ 已重新生成：{CSV_PATH}（{len(rows)} 行 × 30 列）")
    print(f"   SHA-256：{got}")
    print(f"   与组内锚定：{'✅ 一致' if got == CANON_SHA else '⚠️ 不一致（重建后请通知 C 核对 manifest）'}")


if __name__ == "__main__":
    main()
