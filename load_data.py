import os
import pandas as pd
from scipy.io.arff import loadarff

# 1. 定义数据路径
credit_arff_path = 'data/raw/credit-g.arff'
sick_arff_path = 'data/raw/sick.arff'

print("正在加载本地 ARFF 数据...")

# 2. 读取 credit-g
raw_credit, meta_credit = loadarff(credit_arff_path)
df_credit = pd.DataFrame(raw_credit)

# 3. 读取 sick
raw_sick, meta_sick = loadarff(sick_arff_path)
df_sick = pd.DataFrame(raw_sick)

# 4. 转化
for df in [df_credit, df_sick]:
    for col in df.select_dtypes([object]).columns:
        # 将 bytes 解码为 str，并去除前后可能存在的空格
        df[col] = df[col].str.decode('utf-8').str.strip()

# 5. 对 sick 数据集把文本中的 '?' 转为 NaN 缺失值
df_sick.replace('?', pd.NA, inplace=True)

# 6. 打印基本信息确认无误
print("\n--- credit-g 加载成功 ---")
print(f"数据形状 (行, 列): {df_credit.shape}")
print(df_credit.head(2))

print("\n--- sick 加载成功 ---")
print(f"数据形状 (行, 列): {df_sick.shape}")
print(df_sick.head(2))

# 7. 另存为 CSV
os.makedirs('data/raw', exist_ok=True)
df_credit.to_csv('data/raw/credit-g.csv', index=False)
df_sick.to_csv('data/raw/sick.csv', index=False)
print("\n已成功将解码后的数据转存为标准 CSV 格式至 data/raw/ 目录下！")