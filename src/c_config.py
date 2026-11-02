"""
公共配置都放这一个文件: 种子、四个模型的超参、两个数据集的信息。
想改超参数只动这里, 别的代码全部从这里拿。

有一个小坑: B 的 sick 规格单里写的是 "TBG_measured", 但 csv 里实际列名
是 "TBG measured"(中间是空格), 这里按 csv 实际的列名来删。
"""
SEED = 42  # 全组统一用 42

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 四个模型的超参数(两个数据集共用)
# display 是结果表里显示的名字; linear=True 的模型用 OneHot+标准化,
# False 的用 Ordinal 编码+原始数值(树只看排序, 不需要缩放)。
MODELS = {
    "DT": dict(display="DecisionTree", cls=DecisionTreeClassifier,
               kw=dict(criterion="entropy", ccp_alpha=0.01, random_state=SEED),
               linear=False),   # 熵+剪枝, 差不多就是 Weka J48 的做法
    "RF": dict(display="RandomForest", cls=RandomForestClassifier,
               kw=dict(n_estimators=100, random_state=SEED, n_jobs=-1),
               linear=False),
    "LR": dict(display="Logistic", cls=LogisticRegression,
               kw=dict(max_iter=1000, random_state=SEED),
               linear=True),
    "SVC": dict(display="SVC", cls=SVC,
                kw=dict(kernel="rbf", C=1.0, probability=True, random_state=SEED),
                linear=True),   # RBF 核, 对应 Weka 的 SMO
}

# 两个数据集的基本信息
DATASETS = {
    "creditg": dict(
        dataset_d="credit-g",     # 带连字符, 出现在切分文件名里
        file="data/raw/credit-g.csv",
        class_col="class",        # 注意是小写
        pos_label="bad", neg_label="good",
        drop_cols=[],             # 数据干干净净, 不用清洗
        class_weight={"bad": 5, "good": 1},   # 官方 5:1 代价矩阵
        n_rows=1000, n_features=20,              # 21 列 − 目标列
        ratio="700:300",
    ),
    "sick": dict(
        dataset_d="sick",
        file="data/raw/sick.csv",
        class_col="Class",        # 首字母大写! 小写 sick 是个特征列, 别搞混
        pos_label="sick", neg_label="negative",
        drop_cols=["TBG measured", "TBG"],    # TBG 整列全空, 直接在管线外删
        class_weight={"sick": 1, "negative": 1},  # sick 没官方代价, 等权当对照
        n_rows=3772, n_features=27,              # 30 列 − TBG两列 − 目标列
        ratio="3541:231",
    ),
}

STRATEGIES = ["plain", "smote", "cost", "threshold"]  # threshold 的实现在 c4_threshold.py
RUN_ID = "run1"    # 运行编号, 会出现在输出文件名里
