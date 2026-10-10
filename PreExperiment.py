# 在原始数据集上做决策树和10折交叉验证，和cyn同步准确率

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

def evaluate_base(csv_path, target_col):
    print(f"正在评估数据集：{csv_path}")
    df = pd.read_csv(csv_path, na_values="?")
    print(f"原始数据（行，列）：{df.shape}")
    # 分离特征
    if target_col not in df.columns:
        raise ValueError(f"目标列‘{target_col}'不在数据集中，检查列名")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 预处理
    # 1.缺失值：众数填充或直接删掉
    X = X.fillna(X.mode().iloc[0])

    # 2.文本类转为one-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # 3.y一定是纯数值或字符串类型
    y = y.astype(str)

    # decesiontree + 10 折交叉验证
    model = DecisionTreeClassifier(random_state=42)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # 计算Accuracy
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    # 打印结果
    print(f"10折交叉验证每折准确率: {scores}")
    print(f"基线平均准确率（Mean，Accuracy）: {scores.mean():.4f}")
    print(f"准确率标准差（Std）: {scores.std():.4f}")


# 指定目标列，给出决策树答案
# __name__：Python 的一个内置变量，用来记录当前这个文件是怎么被运行的。
# "__main__"：代表“直接运行”。
if __name__ == "__main__":
    # credit-g
    try:
        evaluate_base(csv_path='data/raw/credit-g.csv', target_col='class')
    except Exception as e:
        print("credit-g 运行错误", e)

    # sick
    try:
        evaluate_base(csv_path='data/raw/sick.csv', target_col='Class')
    except Exception as e:
        print("sick 运行错误", e)

