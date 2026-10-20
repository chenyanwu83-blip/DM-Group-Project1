"""
B5 · 打印 sick 统一流水线结构（交 C 嵌入 + C8 防泄漏检查留证）
===============================================================
构建与组长方案 C2/C4 一致的 imblearn Pipeline，把结构 print 出来：
  - 预处理（插补/编码/标准化）→ 可选采样器（SMOTE）→ 分类器
  - 所有统计量（均值/众数/标准差/SMOTE 合成点）都只在训练折 fit，
    验证折只做 transform —— 这就是"防数据泄漏"的机器可读证据。

运行方式：
    conda activate cds535
    python src/b5_print_pipeline.py
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
df.columns = [c.replace(" measured", "_measured") for c in df.columns]  # 兼容组仓库空格版列名
df = df.drop(columns=["TBG", "TBG_measured"])          # B3 决策：全缺列删除
X = df.drop(columns=["Class"])
NUM = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
CAT = [c for c in X.columns if c not in NUM]

# 预处理 A：树模型用（插补 + 序数编码；树不吃 one-hot 也不需要标准化）
pre_tree = ColumnTransformer([
    ("num", SimpleImputer(strategy="mean"), NUM),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                             unknown_value=-1))]), CAT),
], remainder="drop")

# 预处理 B：Logistic / SVC 用（插补 + 标准化 + 独热编码）
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

print("=" * 74)
print("B5 · sick 统一流水线结构（4 模型 × 4 策略 = 16 配置的骨架）")
print("=" * 74)

for name, clf in MODELS.items():
    pre = pre_tree if name in ("DecisionTree", "RandomForest") else pre_linear
    pipe_plain = Pipeline([("pre", pre), ("clf", clf)])
    print(f"\n── {name} · plain ──")
    print(pipe_plain)

print("\n── 加 SMOTE 后的结构（以 Logistic+SMOTE 为例；重采样仅发生在训练折）──")
print(Pipeline([("pre", pre_linear),
                ("smote", SMOTE(random_state=SEED)),
                ("clf", MODELS["Logistic"])]))

print("""
说明（对应 configs/spec_sick.md 规格单）：
  · 删全缺列（TBG、TBG_measured）在 Pipeline 外做——不依赖任何统计量，无泄漏风险
  · 插补 / 编码 / 标准化 / SMOTE 全部在 Pipeline 内 → imblearn 保证它们只 fit 训练折
  · 代价敏感策略：sick 侧按方案用等代价对照 class_weight={'sick':1,'negative':1}
    （结果应≈plain，用作流水线正确性的 sanity check）
  · 阈值移动策略：训练折内再切 80/20 选 τ，锁定后才在测试折用（详见 run_experiment.py）
""")
