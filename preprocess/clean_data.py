import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from scipy import stats

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP']

DATASET_PATH = sys.argv[1]
LABEL_PATH = sys.argv[2]
ZSCORE_THRESHOLD = float(sys.argv[3])

if DATASET_PATH in ['preprocess/bank_marketing_data_X_train_scaled.csv',
                    'preprocess/boston_housing_data_X_train_scaled.csv']:
    DS_TYPE = "no_test"
    if DATASET_PATH == 'preprocess/bank_marketing_data_X_train_scaled.csv':
        DS_NAME = "bank_marketing"
        DISPLAY_NAME = "Bank Marketing"
    elif DATASET_PATH == 'preprocess/boston_housing_data_X_train_scaled.csv':
        DS_NAME = "boston_housing"
        DISPLAY_NAME = "Boston Housing"
elif DATASET_PATH == 'preprocess/breast_cancer_elvira_data_X_{}_scaled.csv':
    DS_TYPE = "w_test"
    DS_NAME = "breast_cancer_elvira"
    DISPLAY_NAME = "Breast Cancer"
else:
    raise ValueError("Invalid dataset path")

def map_categorical_to_range(df):
    """将字符串类型的特征映射到 [-1, 1] 区间"""
    for column in df.select_dtypes(include='object').columns:
        # 对每个字符串特征进行 Label Encoding
        label_encoder = LabelEncoder()
        df[column] = label_encoder.fit_transform(df[column])
        # 将类别编码归一化到 [0, 1]
        df[column] = df[column] / (df[column].max() - df[column].min())
        # 再将其映射到 [-1, 1]
        df[column] = 2 * df[column] - 1
    return df

def visualize_clustering_results(X_2d, labels, title, display_title, xlim=None, ylim=None):
    plt.figure(figsize=(8, 6))
    mask_valid = labels != -1
    scatter_valid = plt.scatter(X_2d[mask_valid, 0], X_2d[mask_valid, 1], 
                                c=labels[mask_valid], cmap='viridis', alpha=0.5)
    mask_invalid = labels == -1
    _ = plt.scatter(X_2d[mask_invalid, 0], X_2d[mask_invalid, 1], 
                                  marker='x', color='gray', alpha=0.5)
    plt.colorbar(scatter_valid)

    plt.title(f'{display_title} 标签分布可视化')
    plt.xlabel('主成分 1')
    plt.ylabel('主成分 2')

    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    plt.savefig(f'preprocess/visualization_{title}.png')
    plt.clf()

def remove_outliers_zscore(X, y, threshold=3.0):
    """使用Z-score方法去除离群点，并同步移除对应的标签，默认Z-score阈值为3"""
    z_scores = stats.zscore(X)
    filter_condition = (abs(z_scores) < threshold).all(axis=1)
    return X[filter_condition], y[filter_condition], (~filter_condition).sum()

if __name__ == '__main__':
    # 加载训练集数据
    if DS_TYPE == "no_test":
        X_train_scaled = pd.read_csv(DATASET_PATH)
    elif DS_TYPE == "w_test":
        X_train_scaled = pd.read_csv(DATASET_PATH.format('train'))

    # 保存原始的列名
    original_columns = X_train_scaled.columns
    # original_columns = X_train_scaled.dropna(axis=1, how='all').columns
    # 自动识别并映射字符串类型特征到 [-1, 1]
    X_train_scaled = map_categorical_to_range(X_train_scaled)

    # 使用 SimpleImputer 填补缺失值
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train_scaled)
    X_train_scaled = pd.DataFrame(X_train_imputed, columns=original_columns)
    
    # 使用训练集训练 PCA 模型
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(X_train_scaled)
    
    # 获取训练集的 x 和 y 轴范围
    xlim = (X_train_2d[:, 0].min(), X_train_2d[:, 0].max())
    ylim = (X_train_2d[:, 1].min(), X_train_2d[:, 1].max())

    # 加载人类标签
    label_encoder = LabelEncoder()
    if DS_TYPE == "no_test":
        y_train = pd.read_csv(LABEL_PATH)
        y_columns = y_train.columns
        y_train = label_encoder.fit_transform(y_train.values.ravel())
    elif DS_TYPE == "w_test":
        y_train = pd.read_csv(LABEL_PATH.format('train'))
        y_columns = y_train.columns
        y_train = label_encoder.fit_transform(y_train.values.ravel())

    X_train_cleaned, y_train_cleaned, outliers_count = remove_outliers_zscore(X_train_scaled, y_train, threshold=ZSCORE_THRESHOLD)
    X_train_2d = pca.transform(X_train_cleaned)
    
    # 打印去除了多少离群点
    print(f"去除了 {outliers_count} 个离群点")
    
    if DS_TYPE == "no_test":
        visualize_clustering_results(X_train_2d, y_train_cleaned, DS_NAME + "_cleaned", DISPLAY_NAME + "数据集", xlim, ylim)
    elif DS_TYPE == "w_test":
        visualize_clustering_results(X_train_2d, y_train_cleaned, DS_NAME + "_train_cleaned", DISPLAY_NAME + "训练集", xlim, ylim)
    
    # 保存清理后的数据和标签
    X_train_cleaned_df = pd.DataFrame(X_train_cleaned, columns=original_columns)
    y_train_cleaned_df = pd.DataFrame(y_train_cleaned, columns=y_columns)
    
    X_train_cleaned_df.to_csv(f'preprocess/{DS_NAME}_cleaned_train_data.csv', index=False)
    y_train_cleaned_df.to_csv(f'preprocess/{DS_NAME}_cleaned_train_labels.csv', index=False)
    
    print(f"清理后的数据已保存到 'preprocess/{DS_NAME}_cleaned_train_data.csv'")
    print(f"清理后的标签已保存到 'preprocess/{DS_NAME}_cleaned_train_labels.csv'")
