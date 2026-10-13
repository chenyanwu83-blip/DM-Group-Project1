# -*- coding: utf-8 -*-
"""C2: 统一流水线配置常量 —— 冻结规格的代码。

依据(2026-10-12 v1.0):
  - configs/spec_creditg.md        (A, credit-g 预处理规格单)
  - configs/spec_sick.md           (B, sick 预处理规格单)
  - docs/D1_评估协议_v1.0.md        (D, 标签/协议/指标口径)
  - docs/D2_给ABC的调用说明.md      (D, 接口与输出命名)

已登记差异(详见 docs/C2_流水线规格回执_v1.0.md 第5节):
  spec_sick.md §1① 写删除 "TBG_measured", 而 data/raw/sick.csv 实际列名为
  "TBG measured"(空格分隔)。本代码按实际 CSV 列名删列, 规格语义不变。
"""
SEED = 42  # 全组锁死

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ---- 4 模型固定超参(两数据集统一, spec §3) ----
# display 填 D2 模板 model 列(D2 示例用 "DecisionTree");
# linear=True 走 OneHot+StandardScaler 通道, False 走 Ordinal+数值 passthrough 通道。
MODELS = {
    "DT": dict(display="DecisionTree", cls=DecisionTreeClassifier,
               kw=dict(criterion="entropy", ccp_alpha=0.01, random_state=SEED),
               linear=False),          # 对应 Weka J48 的熵+剪枝
    "RF": dict(display="RandomForest", cls=RandomForestClassifier,
               kw=dict(n_estimators=100, random_state=SEED, n_jobs=-1),
               linear=False),
    "LR": dict(display="Logistic", cls=LogisticRegression,
               kw=dict(max_iter=1000, random_state=SEED),
               linear=True),
    "SVC": dict(display="SVC", cls=SVC,
                kw=dict(kernel="rbf", C=1.0, probability=True, random_state=SEED),
                linear=True),          # 对应 Weka SMO
}

# ---- 数据集元信息(D1 标签口径 + A/B 冻结预处理) ----
DATASETS = {
    "creditg": dict(
        dataset_d="credit-g",                    # D 口径数据集名(带连字符)
        file="data/raw/credit-g.csv",
        class_col="class",                       # 小写(D1 核查)
        pos_label="bad", neg_label="good",
        drop_cols=[],                            # A3 全项通过(0缺失 0重复) → 零清洗
        class_weight={"bad": 5, "good": 1},      # A5 冻结: 官方 5:1 代价矩阵(FN:FP)
        n_rows=1000, n_features=20,              # 21 列 − 目标列
        ratio="700:300",
    ),
    "sick": dict(
        dataset_d="sick",
        file="data/raw/sick.csv",
        class_col="Class",                       # 首字母大写! 小写 sick 是特征列(D1/D2)
        pos_label="sick", neg_label="negative",
        drop_cols=["TBG measured", "TBG"],       # B5① 管线外删(TBG 100%缺/配对标记恒'f')
        class_weight={"sick": 1, "negative": 1},  # B5 冻结: 等代价对照, 结果应≈plain
        n_rows=3772, n_features=27,              # 30 列 − TBG两列 − 目标列
        ratio="3541:231",
    ),
}

STRATEGIES = ["plain", "smote", "cost", "threshold"]  # threshold 实现待 C4 验证启用
RUN_ID = "run1"                                        # D2 示例口径
