"""
数据读取 + imblearn 流水线, 整个项目的防泄漏就靠这里。

读数据统一 pd.read_csv(path, na_values="?"), 不排序不删行, 原始 csv 一个字不动。
标签全程用原始字符串(bad/good, sick/negative), 不做 0/1 编码,
这样 class_weight 直接用规格里的键, 算指标时 y_true/y_pred 也是同一套东西。

防泄漏的关键: 插补/编码/标准化/SMOTE 全塞在 imblearn 的 Pipeline 里,
交叉验证时只有训练折被 fit, 验证折永远只 transform。
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipe
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline   # 注意是 imblearn 的! sklearn 的 Pipeline 不会在折内做采样

from src.c_config import SEED, MODELS, DATASETS


def load_dataset(short):
    """读入并分离 X/y。short ∈ {creditg, sick}。"""
    m = DATASETS[short]
    df = pd.read_csv(m["file"], na_values="?")   # 读法和 evaluate.py 保持一致
    if m["drop_cols"]:
        df = df.drop(columns=m["drop_cols"])
    assert df.shape == (m["n_rows"], m["n_features"] + 1), \
        f"{short}: 形状 {df.shape} 与规格不符"
    labels = set(df[m["class_col"]].unique())
    assert labels == {m["pos_label"], m["neg_label"]}, \
        f"{short}: 标签 {labels} 与 D1 不符"
    y = df[m["class_col"]]   # 字符串标签一路到底, 不编码
    return df.drop(columns=[m["class_col"]]), y


def _pipe(steps):
    """无组件→passthrough; 单组件直接返回; 多组件才包 sklearn.Pipeline。"""
    if not steps:
        return "passthrough"
    return SkPipe(steps) if len(steps) > 1 else steps[0][1]


def make_pre(short, linear=True):
    """把预处理规格变成 ColumnTransformer。

    linear=True  给 LR/SVC: 名义属性 OneHot + 数值 StandardScaler
    linear=False 给 DT/RF : 名义属性 Ordinal + 数值不缩放(树只看大小关系)
    sick 要先在折内插补(数值用均值/名义用众数), credit-g 干净不用。
    """
    X, _ = load_dataset(short)   # 只是看一眼列类型, 不参与建模
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


def _assemble(short, model_key, strategy):
    """strategy ∈ {plain, smote, cost} → imblearn Pipeline。"""
    m, meta = MODELS[model_key], DATASETS[short]
    clf = m["cls"](**m["kw"])
    if strategy == "cost":
        clf.set_params(class_weight=meta["class_weight"])
    steps = [("pre", make_pre(short, linear=m["linear"]))]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("clf", clf))
    return Pipeline(steps)


def build_pipe(short, model_key, strategy="plain"):
    """统一入口, run_experiment 每折都会来调一次。

    plain    基线, 什么都不加
    smote    SMOTE 放在 Pipeline 里(random_state=42), 所以只在训练折执行
    cost     class_weight 按规格来(credit-g 用 5:1, sick 等权对照)
    threshold 返回 ThresholdClassifier 包装器: 里面包一条 plain 流水线,
             训练时内部再分 5 折选好阈值 tau 并锁定, tau 全程不碰外层验证折
    """
    if strategy == "threshold":
        meta = DATASETS[short]
        from src.c4_threshold import ThresholdClassifier   # 延迟导入避免环
        return ThresholdClassifier(
            make_base=lambda: _assemble(short, model_key, "plain"),
            pos_label=meta["pos_label"], neg_label=meta["neg_label"], seed=SEED)
    return _assemble(short, model_key, strategy)
