"""
打印 sick 数据集的统一 imblearn 流水线结构：
预处理（插补/编码/标准化）、可选 SMOTE、分类器三段。
所有预处理都在 Pipeline 内，只在训练折上拟合，验证折只做转换。
"""
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

df = pd.read_csv(ROOT / "data" / "raw" / "sick.csv", na_values="?")
df.columns = [c.replace(" measured", "_measured") for c in df.columns]
df = df.drop(columns=["TBG", "TBG_measured"])          # 全缺列删除
X = df.drop(columns=["Class"])
NUM = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
CAT = [c for c in X.columns if c not in NUM]

# 树模型的预处理：插补 + 序数编码
pre_tree = ColumnTransformer([
    ("num", SimpleImputer(strategy="mean"), NUM),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                             unknown_value=-1))]), CAT),
], remainder="drop")

# 线性模型的预处理：插补 + 标准化 + 独热编码
pre_linear = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="mean")),
                      ("sc", StandardScaler())]), NUM),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), CAT),
], remainder="drop")

MODELS = {
    "DecisionTree": DecisionTreeClassifier(criterion="entropy", ccp_alpha=0.01,
                                           random_state=SEED),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=SEED),
    "Logistic": LogisticRegression(max_iter=1000, random_state=SEED),
    "SVC": SVC(kernel="rbf", C=1.0, probability=True, random_state=SEED),
}

print("sick 统一流水线结构")

for name, clf in MODELS.items():
    pre = pre_tree if name in ("DecisionTree", "RandomForest") else pre_linear
    pipe_plain = Pipeline([("pre", pre), ("clf", clf)])
    print(f"\n── {name} · plain ──")
    print(pipe_plain)

print("\n加 SMOTE 后的结构（以 Logistic 为例，重采样只在训练折）")
print(Pipeline([("pre", pre_linear),
                ("smote", SMOTE(random_state=SEED)),
                ("clf", MODELS["Logistic"])]))

print("""
说明：
  删全缺列在 Pipeline 外做，不依赖统计量，无泄漏风险。
  插补/编码/标准化/SMOTE 都在 Pipeline 内，只在训练折上拟合。
  代价敏感策略用等权重对照 class_weight={'sick':1,'negative':1}，结果应与 plain 一致。
  阈值移动策略：在训练折内选阈值，锁定后再用于测试（见 run_experiment.py）。
""")
