# -*- coding: utf-8 -*-
"""C2: 统一 imblearn 流水线(防泄漏核心)。

数据读取与标签口径(D1):
  - 统一 pd.read_csv(path, na_values="?"), 不排序、不删行、不重写原始 CSV;
  - y 全程保留原始字符串标签(bad/good, sick/negative), 不做 0/1 编码
    —— class_weight 可用冻结规格原键, 送指标库时 y_true/y_pred/classes_ 同源同类型;
  - sick 仅删规格指定两列(TBG 系), 在管线外(列删除不依赖统计量, 无泄漏)。

防泄漏(D1 cv10 第4条):
  插补/编码/标准化/重采样全部在 imblearn.pipeline.Pipeline 内,
  交叉验证只在训练折 fit, 验证折一律只 transform。
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipe
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline              # 不是 sklearn.pipeline! 采样器必须由它承载

from src.c_config import SEED, MODELS, DATASETS


def load_dataset(short):
    """读入并分离 X/y。short ∈ {creditg, sick}。"""
    m = DATASETS[short]
    df = pd.read_csv(m["file"], na_values="?")      # 与 src/evaluate.py 完全同款读法
    if m["drop_cols"]:
        df = df.drop(columns=m["drop_cols"])
    assert df.shape == (m["n_rows"], m["n_features"] + 1), \
        f"{short}: 形状 {df.shape} 与规格不符"
    labels = set(df[m["class_col"]].unique())
    assert labels == {m["pos_label"], m["neg_label"]}, \
        f"{short}: 标签 {labels} 与 D1 不符"
    y = df[m["class_col"]]                          # 原始字符串标签, 全程不编码
    return df.drop(columns=[m["class_col"]]), y


def _pipe(steps):
    """无组件→passthrough; 单组件直接返回; 多组件才包 sklearn.Pipeline。"""
    if not steps:
        return "passthrough"
    return SkPipe(steps) if len(steps) > 1 else steps[0][1]


def make_pre(short, linear=True):
    """冻结规格 → ColumnTransformer。规格改版只动本函数, 流水线结构不动。

    linear=True : A5②/B5④ 线性通道(LR/SVC)  名义 OneHot + 数值 StandardScaler
    linear=False: A5③/B5③ 树通道(DT/RF)      名义 Ordinal + 数值 passthrough 不缩放
    sick 侧先折内插补(B5②: 数值 mean / 名义 most_frequent); credit-g 零清洗(A5①)。
    """
    X, _ = load_dataset(short)                      # 仅为探列类型, 不入模型
    num = X.select_dtypes("number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    if short == "sick":
        cat_steps = [("imp", SimpleImputer(strategy="most_frequent"))]
        num_steps = [("imp", SimpleImputer(strategy="mean"))]
    else:
        cat_steps, num_steps = [], []
    if linear:
        cat_steps.append(("oh", OneHotEncoder(handle_unknown="ignore")))
        num_steps.append(("sc", StandardScaler()))
    else:
        cat_steps.append(("oe", OrdinalEncoder(handle_unknown="use_encoded_value",
                                               unknown_value=-1)))
    return ColumnTransformer([("cat", _pipe(cat_steps), cat),
                              ("num", _pipe(num_steps), num)])


def build_pipe(short, model_key, strategy="plain"):
    """组装: 预处理 → (可选 SMOTE) → 分类器。

    strategy:
      plain    无附加(基线)
      smote    Pipeline 内 SMOTE(random_state=42), 仅训练折执行(imblearn 保证)
      cost     class_weight 用冻结规格原键(credit-g 5:1 / sick 等代价对照)
      threshold 不在此拼装 —— 需"内层5折OOF选τ"的包装器, 见 C2 回执/指南第六节, C4 验证后启用
    """
    m, meta = MODELS[model_key], DATASETS[short]
    clf = m["cls"](**m["kw"])
    if strategy == "cost":
        clf.set_params(class_weight=meta["class_weight"])
    steps = [("pre", make_pre(short, linear=m["linear"]))]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("clf", clf))
    return Pipeline(steps)
