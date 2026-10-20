"""E4: reproduce C's credit-g plain holdout trees and export course-level explanations.

Run from the repository root with the Python environment in requirements.txt.
The plain strategy has no sampler, so sklearn Pipeline is equivalent to C's
imblearn Pipeline here. Exact saved holdout predictions are checked below.
"""

from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import export_text

from src.c_config import DATASETS, MODELS


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXPECTED_SHA256 = "d0baf9fddd41e5a6af0ba84e6037a415f11384dd40c494186563a2d3b68f5c25"


def main():
    data_path = ROOT / DATASETS["creditg"]["file"]
    actual_sha = sha256(data_path.read_bytes()).hexdigest()
    assert actual_sha == EXPECTED_SHA256, "credit-g input changed"
    data = pd.read_csv(data_path, na_values="?")
    assert data.shape == (1000, 21) and not data.isna().any().any()
    X, y = data.drop(columns="class"), data["class"]
    assert y.value_counts().to_dict() == {"good": 700, "bad": 300}
    train = pd.read_csv(ROOT / "results/train_idx_creditg.csv")["row_idx"].to_numpy()
    holdout = pd.read_csv(ROOT / "results/holdout_idx_creditg.csv")["row_idx"].to_numpy()
    assert len(train) == 700 and len(holdout) == 300
    assert len(set(train) | set(holdout)) == 1000 and not set(train) & set(holdout)

    numeric = X.select_dtypes("number").columns.tolist()
    categorical = [column for column in X if column not in numeric]
    # Matches src/c_pipelines.py: tree channel, categorical first, numeric last.
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical),
        ("num", "passthrough", numeric),
    ])
    names = [f"cat__{name}" for name in categorical] + [f"num__{name}" for name in numeric]
    lines = ["credit-g plain holdout model, trained on the locked 700 training rows.",
             f"Input SHA-256: {actual_sha}",
             "Category numbers below are OrdinalEncoder codes, not ordered meanings.", ""]

    for key in ("DT", "RF"):
        model = MODELS[key]
        pipe = Pipeline([("pre", pre), ("clf", model["cls"](**model["kw"]))])
        pipe.fit(X.iloc[train], y.iloc[train])
        saved = pd.read_csv(ROOT / "results/preds" /
                            f"creditg_{key.lower()}_plain_v1_run1_holdout_preds.csv")
        assert saved["row_idx"].tolist() == holdout.tolist()
        assert saved["y_true"].tolist() == y.iloc[holdout].tolist()
        predicted = pipe.predict(X.iloc[holdout])
        assert predicted.tolist() == saved["y_pred"].tolist(), f"{key} predictions differ"
        bad_col = list(pipe.classes_).index("bad")
        probabilities = pipe.predict_proba(X.iloc[holdout])[:, bad_col]
        assert np.allclose(probabilities, saved["proba_pos"].to_numpy(), atol=1e-12), \
            f"{key} probabilities differ"
        lines.append(f"{key}: 300/300 saved holdout labels and probabilities matched.")
        if key == "DT":
            tree = pipe.named_steps["clf"]
            lines.append(f"Decision tree: {tree.get_depth()} levels, {tree.get_n_leaves()} leaves.")
            lines.append(f"Classes: {list(tree.classes_)}")
            lines.append(export_text(tree, feature_names=names, decimals=2, show_weights=True))
            for field, labels in zip(categorical, pipe.named_steps["pre"].named_transformers_["cat"].categories_):
                if field in ("checking_status", "purpose", "other_parties", "employment"):
                    lines.append(f"{field} codes: " + ", ".join(f"{i}={label}" for i, label in enumerate(labels)))
        else:
            importance = pd.DataFrame({"feature": categorical + numeric,
                                       "importance": pipe.named_steps["clf"].feature_importances_})
            importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
            importance.insert(0, "rank", importance.index + 1)
            assert np.isclose(importance["importance"].sum(), 1.0)
            importance.to_csv(OUT / "e4_rf_feature_importance.csv", index=False)
    (OUT / "e4_decision_tree_paths.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
