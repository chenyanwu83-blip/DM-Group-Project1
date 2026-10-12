"""Prepare a reproducible Boolean transaction matrix for credit-g Apriori.

Run with pandas installed. This script never changes the raw input file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.preprocessing import KBinsDiscretizer


ATTRIBUTE = re.compile(r"^@attribute\s+(\S+)\s+(.+)$", re.IGNORECASE)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_arff(path: Path) -> tuple[pd.DataFrame, list[str]]:
    names: list[str] = []
    numeric: list[str] = []
    rows: list[list[str]] = []
    in_data = False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        match = ATTRIBUTE.match(line)
        if match:
            name, kind = match.groups()
            names.append(name)
            if kind.lower() in {"real", "numeric", "integer"}:
                numeric.append(name)
            continue
        if line.lower() == "@data":
            in_data = True
            continue
        if in_data:
            rows.append(next(csv.reader([line], quotechar="'", skipinitialspace=True)))
    if not names or any(len(row) != len(names) for row in rows):
        raise ValueError("ARFF header or row width is invalid")
    frame = pd.DataFrame(rows, columns=names)
    for name in numeric:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    return frame, numeric


def load_input(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if path.suffix.lower() == ".arff":
        return load_arff(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        numeric = frame.select_dtypes(include="number").columns.tolist()
        return frame, numeric
    raise ValueError("Input must be .arff or .csv")


def prepare(input_path: Path, spec_path: Path, output_dir: Path) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    input_hash = sha256(input_path)
    if input_hash != spec["source_sha256"]:
        raise ValueError(f"Input SHA-256 does not match A's audited source: {input_hash}")
    a5_path = spec_path.parent / spec["a5_spec_file"]
    if sha256(a5_path) != spec["a5_spec_sha256"]:
        raise ValueError("Local A5 specification has changed; review it before rerunning")
    clarification_path = spec_path.parent / spec["a5_clarification_file"]
    if sha256(clarification_path) != spec["a5_clarification_sha256"]:
        raise ValueError("Local A5 clarification has changed; review it before rerunning")
    frame, numeric_columns = load_input(input_path)
    class_column = spec["class_column"]
    required = {class_column, *spec["quantile_bins"], *spec["integer_category_columns"]}
    absent = sorted(required - set(frame.columns))
    if absent:
        raise ValueError(f"Input columns do not match the specification: {absent}")
    if frame.isna().any().any() or (frame.astype(str) == "?").any().any():
        raise ValueError("Unexpected missing values: decide handling before Apriori")
    if len(frame) != 1000:
        raise ValueError(f"Expected 1000 credit-g records; got {len(frame)}")
    if frame.shape[1] != 21 or frame.duplicated().any():
        raise ValueError("Expected 21 original columns and no duplicate records")
    labels = frame[class_column].astype(str).value_counts().to_dict()
    if labels != {"good": 700, "bad": 300}:
        raise ValueError(f"Unexpected class counts: {labels}")

    transformed = frame.copy()
    columns_to_bin = list(spec["quantile_bins"])
    requested_bins = list(spec["quantile_bins"].values())
    numeric_to_bin = frame[columns_to_bin].apply(pd.to_numeric, errors="raise")
    discretizer = KBinsDiscretizer(
        n_bins=requested_bins, encode="ordinal", strategy="quantile"
    )
    codes = discretizer.fit_transform(numeric_to_bin)
    bin_log = {}
    for index, column in enumerate(columns_to_bin):
        actual_bins = int(discretizer.n_bins_[index])
        if actual_bins != requested_bins[index]:
            raise ValueError(f"A5 requests {requested_bins[index]} bins for {column}, got {actual_bins}")
        transformed[column] = [f"q{int(code) + 1}_of_{actual_bins}" for code in codes[:, index]]
        bin_log[column] = {
            "method": "sklearn.preprocessing.KBinsDiscretizer",
            "requested_bins": requested_bins[index],
            "actual_bins": actual_bins,
            "edges": [float(edge) for edge in discretizer.bin_edges_[index]],
            "counts": transformed[column].value_counts().sort_index().to_dict(),
        }

    remaining_numeric = set(numeric_columns) - set(spec["quantile_bins"])
    if remaining_numeric != set(spec["integer_category_columns"]):
        raise ValueError(
            "Every numeric column must be binned or explicitly treated as a category. "
            f"Unmatched: {sorted(remaining_numeric ^ set(spec['integer_category_columns']))}"
        )
    for column in spec["integer_category_columns"]:
        transformed[column] = pd.to_numeric(
            transformed[column], errors="raise"
        ).map(lambda value: str(int(value)) if float(value).is_integer() else str(value))

    categorical = transformed.astype(str)
    transactions = pd.get_dummies(categorical, prefix_sep="=", dtype=bool)
    if not transactions.any(axis=1).all():
        raise ValueError("At least one row has no items")
    for column in categorical:
        group = [name for name in transactions if name.startswith(f"{column}=")]
        if not transactions[group].sum(axis=1).eq(1).all():
            raise ValueError(f"One-hot encoding invalid for {column}")
    if int(transactions[f"{class_column}=bad"].sum()) != 300:
        raise ValueError("Positive-class indicator count changed")

    output_dir.mkdir(parents=True, exist_ok=True)
    discrete_path = output_dir / "credit_g_discretized.csv"
    categorical.to_csv(discrete_path, index=False)
    matrix_path = output_dir / "credit_g_transactions.csv"
    transactions.to_csv(matrix_path, index=False)
    metadata = {
        "status": spec["status"],
        "input_path": str(input_path.resolve()),
        "input_sha256": input_hash,
        "spec_path": str(spec_path.resolve()),
        "spec_sha256": sha256(spec_path),
        "a5_commit": spec["a5_commit"],
        "a5_spec_sha256": spec["a5_spec_sha256"],
        "a5_clarification_sha256": spec["a5_clarification_sha256"],
        "sklearn_version": sklearn.__version__,
        "rows": len(transactions),
        "source_columns": list(frame.columns),
        "numeric_columns": numeric_columns,
        "transaction_items": len(transactions.columns),
        "class_counts": labels,
        "binning": bin_log,
        "integer_category_columns": spec["integer_category_columns"],
        "integer_category_policy": spec["integer_category_policy"],
        "integer_category_counts": {
            column: categorical[column].value_counts().sort_index().to_dict()
            for column in spec["integer_category_columns"]
        },
        "true_items_per_row_min": int(transactions.sum(axis=1).min()),
        "true_items_per_row_max": int(transactions.sum(axis=1).max()),
        "discretized_path": str(discrete_path.resolve()),
        "discretized_sha256": sha256(discrete_path),
        "output_path": str(matrix_path.resolve()),
        "output_sha256": sha256(matrix_path),
        "note": "One row per record; each original column contributes exactly one True item.",
    }
    (output_dir / "conversion_record.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "rows": metadata["rows"],
        "transaction_items": metadata["transaction_items"],
        "class_counts": labels,
        "actual_bins": {key: item["actual_bins"] for key, item in bin_log.items()},
        "discretized": str(discrete_path),
        "output": str(matrix_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.input, args.spec, args.output_dir)
