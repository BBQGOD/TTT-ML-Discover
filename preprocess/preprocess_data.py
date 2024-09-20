import re
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

DATASET_PATH = sys.argv[1]
DS_TYPE = None
if DATASET_PATH == 'bank_marketing_data/bank-full.csv':
    DS_TYPE = "csv_semicolon"
    DISPLAY_NAME = "Bank Marketing"
elif DATASET_PATH == 'boston_housing_data/HousingData.csv':
    DS_TYPE = "csv_comma"
    DISPLAY_NAME = "Boston Housing"
elif DATASET_PATH == "breast_cancer_elvira_data":
    DS_TYPE = "dir"
    DISPLAY_NAME = "Breast Cancer"
else:
    raise ValueError("Invalid dataset path")
DS_NAME = DATASET_PATH.split('/')[0]

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']

if DS_TYPE == "csv_semicolon":
    data = pd.read_csv(DATASET_PATH, delimiter=';')
    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]
elif DS_TYPE == "csv_comma":
    data = pd.read_csv(DATASET_PATH)
    X_data = data.iloc[:, :-1]
    y_data = data.iloc[:, -1]
elif DS_TYPE == "dir":
    
    def load_dbc(ds_path):
        with open(ds_path, 'r') as f:
            lines = f.readlines()

        # 提取特征信息
        feature_names = []
        continuous_features = []
        data_start = False
        for line in lines:
            line = line.strip()

            if line.startswith("node"):
                # 提取特征名
                feature_match = re.match(r"node\s+(\S+)\s+\((\w+)\)", line)
                if feature_match:
                    feature_name = feature_match.group(1)
                    feature_type = feature_match.group(2)
                    feature_names.append(feature_name)
                    continuous_features.append(feature_type == "continuous")

            elif line.startswith("relation"):
                # 数据部分开始
                data_start = True
                break

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

        assert len(data) > 0, "数据文件中没有样本数据，请检查文件格式"

        # 转换为 pandas DataFrame
        data = pd.DataFrame(data, columns=['SampleType'] + feature_names)

        # 将连续特征转换为浮点数
        for i, is_continuous in enumerate(continuous_features):
            if is_continuous:
                data.iloc[:, i + 1] = pd.to_numeric(data.iloc[:, i + 1], errors='coerce')

        # 处理缺失值（问号）
        data.replace('?', np.nan, inplace=True)

        # 分离特征和标签
        X_data = data.iloc[:, 1:]  # 特征
        y_data = data.iloc[:, 0]   # 类别标签

        assert len(X_data) == len(y_data), "特征和标签的行数不匹配"
        return X_data, y_data, continuous_features

    X_train, y_train, continuous_features_train = load_dbc(DATASET_PATH + "/breastCancer_Train.dbc")
    X_test, y_test, continuous_features_test = load_dbc(DATASET_PATH + "/breastCancer_Test.dbc")

# 初始化 StandardScaler
scaler = StandardScaler()

# 定义函数来可视化均值和标准差并分别应用之前的坐标轴范围
def plot_mean_std(X_, plot_name):
    dataset_name = "训练集" if 'train' in plot_name else "测试集"
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(X_.mean(axis=0, numeric_only=True))
    plt.title(f'{DISPLAY_NAME} {dataset_name}平均值分布图')
    plt.xlabel('特征')
    plt.ylabel('平均值')
    if 'train' in plot_name:
        plt.ylim(-1, 1)

    plt.subplot(1, 2, 2)
    plt.plot(X_.std(axis=0, numeric_only=True))
    plt.title(f'{DISPLAY_NAME} {dataset_name}标准差分布图')
    plt.xlabel('特征')
    plt.ylabel('标准差')
    if 'train' in plot_name:
        plt.ylim(0, 2)
    
    plt.tight_layout()
    plt.savefig(f'preprocess/mean_std_{plot_name}.png')
    plt.clf()

def scale_continuous_features(X, continuous_mask, fit_scaler=False):
    continuous_columns = X.columns[continuous_mask]
    X_continuous = X[continuous_columns]
    X_discrete = X.loc[:, ~continuous_mask]
    
    # 对连续特征进行归一化
    if fit_scaler:
        X_continuous_scaled = scaler.fit_transform(X_continuous)  # 训练集上 fit 并 transform
    else:
        X_continuous_scaled = scaler.transform(X_continuous)  # 测试集上仅 transform

    # 将归一化后的连续特征和离散特征合并
    X_scaled = pd.concat([pd.DataFrame(X_continuous_scaled, columns=continuous_columns), X_discrete.reset_index(drop=True)], axis=1)
    
    return X_scaled

# 自动检测离散和连续特征（判断是否为字符串类型）
def detect_continuous_features(X):
    return np.array([pd.api.types.is_numeric_dtype(X[col]) for col in X.columns])

def drop_all_null_columns_synchronously_with_mask(X_train, X_test, continuous_features_train, continuous_features_test):
    # 获取训练集和测试集中各自所有为空的列
    train_null_columns = X_train.columns[X_train.isnull().all()]
    test_null_columns = X_test.columns[X_test.isnull().all()]
    
    # 合并训练集和测试集的所有为空的列，确保这些列在两个数据集中都被删除
    all_null_columns = train_null_columns.union(test_null_columns)
    
    # 获取这些列的索引位置，用于同步删除 continuous_features 掩码中的对应位置
    all_null_indices = [X_train.columns.get_loc(col) for col in all_null_columns]
    
    # 删除训练集和测试集中这些列
    X_train_cleaned = X_train.drop(columns=all_null_columns)
    X_test_cleaned = X_test.drop(columns=all_null_columns)
    
    # 同步删除 continuous_features 中对应的索引
    continuous_features_train_cleaned = np.delete(continuous_features_train, all_null_indices)
    continuous_features_test_cleaned = np.delete(continuous_features_test, all_null_indices)
    
    return X_train_cleaned, X_test_cleaned, continuous_features_train_cleaned, continuous_features_test_cleaned

if DS_TYPE == "dir":
    X_train, X_test, continuous_features_train, continuous_features_test = drop_all_null_columns_synchronously_with_mask(
        X_train, X_test, continuous_features_train, continuous_features_test
    )
    
    print(f"删除全为空的特征后，训练集的形状: {X_train.shape}")
    print(f"删除全为空的特征后，测试集的形状: {X_test.shape}")
    print(f"更新后的 continuous_features_train 掩码长度: {len(continuous_features_train)}")
    print(f"更新后的 continuous_features_test 掩码长度: {len(continuous_features_test)}")

    continuous_mask_train = np.array(continuous_features_train)
    continuous_mask_test = np.array(continuous_features_test)

    # 训练集归一化并拟合scaler
    X_train_scaled = scale_continuous_features(X_train, continuous_mask_train, fit_scaler=True)
    # 测试集使用训练集的scaler进行归一化
    X_test_scaled = scale_continuous_features(X_test, continuous_mask_test, fit_scaler=False)

    print("训练集归一化后的数据形状：", X_train_scaled.shape)
    print("测试集归一化后的数据形状：", X_test_scaled.shape)
    print("训练集归一化后的数据描述性统计：", X_train_scaled.describe())
    print("测试集归一化后的数据描述性统计：", X_test_scaled.describe())
    print("训练集归一化后的缺失值数量:", X_train_scaled.isnull().sum().sum())
    print("测试集归一化后的缺失值数量:", X_test_scaled.isnull().sum().sum())

    plot_mean_std(X_train_scaled, f'{DS_NAME}_train_scaled')
    plot_mean_std(X_test_scaled, f'{DS_NAME}_test_scaled')

    # 将归一化后的数据保存到文件
    X_train_scaled.to_csv(f'preprocess/{DS_NAME}_X_train_scaled.csv', index=False, header=True)
    X_test_scaled.to_csv(f'preprocess/{DS_NAME}_X_test_scaled.csv', index=False, header=True)

    y_train.to_csv(f'preprocess/{DS_NAME}_y_train.csv', index=False, header=True)
    y_test.to_csv(f'preprocess/{DS_NAME}_y_test.csv', index=False, header=True)

else:
    X_data = X_data.dropna(axis=1, how='all')
    print(f"删除全为空的特征后，训练集的形状: {X_data.shape}")

    # 自动检测哪些特征是连续的
    continuous_mask = detect_continuous_features(X_data)
    
    X_train_scaled = scale_continuous_features(X_data, continuous_mask, fit_scaler=True)
    
    print("训练集归一化后的数据形状：", X_train_scaled.shape)
    print("训练集归一化后的数据描述性统计：", X_train_scaled.describe())
    print("训练集归一化后的缺失值数量:", X_train_scaled.isnull().sum().sum())
    
    plot_mean_std(X_train_scaled, f'{DS_NAME}_train_scaled')
    
    # 将归一化后的数据保存到文件
    X_train_scaled.to_csv(f'preprocess/{DS_NAME}_X_train_scaled.csv', index=False, header=True)
    y_data.to_csv(f'preprocess/{DS_NAME}_y_train.csv', index=False, header=True)
