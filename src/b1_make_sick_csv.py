"""
生成或校验 data/raw/sick.csv，并计算 SHA-256。
默认只校验现有文件的哈希是否与 results/split_manifest.json 一致，不重写数据。
加 --force 参数时重新生成：优先用本地 Weka 自带 sick.arff 转换，
否则从 UCI 官网下载 zip，拼接 sick.data 和 sick.test（ucimlrepo 不支持该数据集）。
"""
import csv
import hashlib
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # 项目根目录
RAW_DIR = ROOT / "data" / "raw"
CSV_PATH = RAW_DIR / "sick.csv"

# 列名与组内 load_data.py 一致
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
    print(f"使用本地 arff：{arff_path}")
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
    print(f"未找到本地 arff，从 UCI 下载：{url}")
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
    # 默认只做校验
    if "--force" not in sys.argv and CSV_PATH.exists():
        got = sha256_of(CSV_PATH)
        n_rows = sum(1 for _ in CSV_PATH.open()) - 1
        ok = got == CANON_SHA
        print("数据校验")
        print(f"文件：{CSV_PATH}")
        print(f"规模：{n_rows} 行 × 30 列")
        print(f"SHA-256：{got}")
        print(f"与 split_manifest 记录{'一致' if ok else '不一致'}")
        if not ok:
            print("数据被改动过，可用 git checkout -- data/raw/sick.csv 恢复，")
            print("或运行 python src/b1_make_sick_csv.py --force 重新生成")
        return

    # 重新生成
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    rows = from_weka_arff() or from_uci_zip()
    assert len(rows) == 3772, f"行数异常：{len(rows)}（应为 3772）"
    assert all(len(r) == 30 for r in rows), "存在列数不为 30 的行"

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)

    got = sha256_of(CSV_PATH)
    print(f"已重新生成：{CSV_PATH}（{len(rows)} 行 × 30 列）")
    print(f"SHA-256：{got}")
    print(f"与 split_manifest 记录{'一致' if got == CANON_SHA else '不一致，需通知 C 核对 manifest'}")


if __name__ == "__main__":
    main()
