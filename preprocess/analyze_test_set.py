import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATASET_PATH = sys.argv[1]
DS_TYPE = None
if DATASET_PATH == "breast_cancer_elvira_data":
    DS_TYPE = "dir"
else:
    raise ValueError("Invalid dataset path")
DS_NAME = DATASET_PATH.split('/')[0]
if DS_NAME == "breast_cancer_elvira_data":
    DS_NAME = "breast_cancer_elvira_test"

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']


if DS_TYPE == "dir":
    with open(DATASET_PATH + '/breastCancer_Test.dbc', 'r') as f:
        lines = f.readlines()

    # 提取特征信息
    feature_names = []
    continuous_features = []
    data_start = False
    for line in lines:
        line = line.strip()

        if line.startswith("node"):
            # 提取特征名
            feature_match = re.match(r"node\s+(\S+)\s+\((\w+)\)", line) # node SampleType(finite-states) 不会被提取
            if feature_match:
                feature_name = feature_match.group(1)
                feature_type = feature_match.group(2)
                feature_names.append(feature_name)
                continuous_features.append(feature_type == "continuous")

        elif line.startswith("relation"):
            # 数据部分开始
            data_start = True
            break

    # 安全性检查：确保至少有一个特征
    assert len(feature_names) > 0, "没有特征被提取出来，请检查数据文件格式"

    # 提取样本数据
    data = []
    for line in lines:
        if data_start and line.startswith("["):
            line = re.sub(r"[\[\]]", "", line)
            line = re.sub(r"\s+", "", line)  # 去掉多余的空格
            values = line.split(",")  # 不要使用 ", "，直接用 ","
            assert len(values) == len(feature_names) + 1, "数据行中的特征数量与字段数量不匹配：数据行中有 %d 个特征，字段中有 %d 个特征" % (len(values), len(feature_names))
            data.append(values)

    # 安全性检查：确保数据非空
    assert len(data) > 0, "数据文件中没有样本数据，请检查文件格式"

    # 转换为 pandas DataFrame
    data = pd.DataFrame(data, columns=['Class'] + feature_names)

    # 将连续特征转换为浮点数
    discrete_cnt, continuous_cnt = 0, 0
    for i, is_continuous in enumerate(continuous_features):
        if is_continuous:
            data.iloc[:, i + 1] = pd.to_numeric(data.iloc[:, i + 1], errors='coerce')
            continuous_cnt += 1
        else:
            discrete_cnt += 1
    print(f"连续特征数量: {continuous_cnt}, 离散特征数量: {discrete_cnt}")

    # 处理缺失值（问号）
    data.replace('?', np.nan, inplace=True)

    # 分离特征和标签
    X_data = data.iloc[:, 1:]  # 特征
    y_data = data.iloc[:, 0]   # 类别标签

    # 安全性检查：确保特征和标签数量相同
    assert len(X_data) == len(y_data), "特征和标签的行数不匹配"
    
# 检查缺失值
print("X_test 中的缺失值数量:", X_data.isnull().sum().sum())
print("X_test 中的缺失列数量:", X_data.isnull().all().sum())
print("y_test 中的缺失值数量:", y_data.isnull().sum())
assert len(y_data) == len(X_data)

# 描述性统计
print(X_data.describe())
print(y_data.describe())

# 可视化均值和方差
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(X_data.mean(axis=0, numeric_only=True))
plt.title('Breast Cancer 测试集平均值分布图')
plt.xlabel('特征')
plt.ylabel('平均值')

plt.subplot(1, 2, 2)
plt.plot(X_data.std(axis=0, numeric_only=True))
plt.title('Breast Cancer 测试集标准差分布图')
plt.xlabel('特征')
plt.ylabel('标准差')
plt.tight_layout()
plt.savefig(f'preprocess/{DS_NAME}_mean_std_train.png')
plt.clf()

# 查重
print("X_test 中的重复值数量:", X_data.duplicated().sum())

y_train_counts = y_data.value_counts().sort_index()
x_labels = y_train_counts.index
plt.figure(figsize=(8, 6))
y_train_counts.plot(kind='bar', color='c', alpha=0.7)
plt.title('Breast Cancer 测试集标签分布')
plt.xlabel('类别标签')
plt.ylabel('数量')
plt.xticks(ticks=range(len(x_labels)), labels=x_labels, ha='center', rotation=0)
plt.tight_layout()
plt.savefig(f'preprocess/{DS_NAME}_label_distribution.png')
plt.show()
