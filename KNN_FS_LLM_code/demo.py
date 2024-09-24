import os
import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.neighbors import NearestNeighbors
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rrl-DM_HW'))
from rrl.utils import read_csv

DATA_DIR = '../rrl-DM_HW/dataset'
EARLY_STOP = 50
CLASS_NUM = 2


class DBEncoder:
    """Encoder used for data discretization and binarization."""

    def __init__(self, f_df, discrete=False, y_one_hot=True, drop='first'):
        self.f_df = f_df
        self.discrete = discrete
        self.y_one_hot = y_one_hot
        self.label_enc = preprocessing.OneHotEncoder(categories='auto') if y_one_hot else preprocessing.LabelEncoder()
        # Only initialize OneHotEncoder if not using raw discrete features
        if not discrete:
            self.feature_enc = preprocessing.OneHotEncoder(categories='auto', drop=drop)
        self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.X_fname = None
        self.y_fname = None
        self.discrete_flen = None
        self.continuous_flen = None
        self.mean = None
        self.std = None

    def split_data(self, X_df):
        discrete_data = X_df[self.f_df.loc[self.f_df[1] == 'discrete', 0]]
        continuous_data = X_df[self.f_df.loc[self.f_df[1] == 'continuous', 0]]
        if not continuous_data.empty:
            continuous_data = continuous_data.replace(to_replace=r'.*\?.*', value=np.nan, regex=True)
            continuous_data = continuous_data.astype(float)
        return discrete_data, continuous_data

    def fit(self, X_df, y_df):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        self.label_enc.fit(y_df)
        self.y_fname = list(self.label_enc.get_feature_names_out(y_df.columns)) if self.y_one_hot else y_df.columns

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if do not discretize them.
            self.imp.fit(continuous_data.values)
        
        if not discrete_data.empty and not self.discrete:
            # Fit One-Hot Encoder if discrete=False
            self.feature_enc.fit(discrete_data)
            feature_names = discrete_data.columns
            self.X_fname = list(self.feature_enc.get_feature_names_out(feature_names))
            self.discrete_flen = len(self.X_fname)
        else:
            self.X_fname = list(discrete_data.columns) if not discrete_data.empty else []
            self.discrete_flen = discrete_data.shape[1]

        if not self.discrete:
            self.X_fname.extend(continuous_data.columns)
        else:
            if continuous_data.shape[1] > 0:
                self.X_fname.extend(continuous_data.columns)
        self.continuous_flen = continuous_data.shape[1]

    def transform(self, X_df, y_df, normalized=False, keep_stat=False):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        # Encode string value to int index.
        y = self.label_enc.transform(y_df.values.reshape(-1, 1))
        if self.y_one_hot:
            y = y.toarray()

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if we do not discretize them.
            continuous_data = pd.DataFrame(self.imp.transform(continuous_data.values),
                                           columns=continuous_data.columns)
            if normalized:
                if keep_stat:
                    self.mean = continuous_data.mean()
                    self.std = continuous_data.std()
                continuous_data = (continuous_data - self.mean) / self.std
        
        if not discrete_data.empty:
            if not self.discrete:
                # One-hot encoding if discrete=False
                discrete_data = self.feature_enc.transform(discrete_data)
                X_df = pd.concat([pd.DataFrame(discrete_data.toarray()), continuous_data], axis=1)
            else:
                # Directly use discrete values as category features if discrete=True
                X_df = pd.concat([discrete_data.reset_index(drop=True), continuous_data.reset_index(drop=True)], axis=1)
        else:
            X_df = continuous_data

        return X_df.values, y

def get_data_loader(dataset):
    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)

    db_enc = DBEncoder(f_df, discrete=True)
    db_enc.fit(X_df, y_df)

    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)

    return X, y, db_enc

# 假设 glove_dict 是你的 GloVe 词向量字典
def load_glove_embeddings(glove_file):
    glove_dict = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_dict[word] = vector
    return glove_dict

# 处理离散特征，转换为 GloVe 词向量
def transform_discrete_to_glove(X_inp, discrete_columns, glove_dict, embedding_size=50):
    glove_embeddings = []
    for cid, col in enumerate(discrete_columns):
        embeddings = []
        for value in X_inp[:, cid]:
            if value.lower() in glove_dict:
                embeddings.append(glove_dict[value.lower()])
            else:
                embeddings.append(np.zeros(embedding_size))  # 如果词不在glove中，使用全0向量
        glove_embeddings.append(np.array(embeddings))
    
    # 将所有离散特征的词向量拼接起来
    return np.hstack(glove_embeddings)

# 获取最近的 K 个邻居
def knn_find_k_nearest(X, k=5):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    return distances, indices

# 主函数
def main():
    # 载入数据
    X, y, db_enc = get_data_loader('bank-marketing')
    X = X[:10, :]

    # 加载 GloVe 词向量
    glove_file = 'glove.6B.50d.txt'  # 例如 'glove.6B.50d.txt'
    glove_dict = load_glove_embeddings(glove_file)

    # 获取离散和连续特征
    discrete_columns = db_enc.f_df.loc[db_enc.f_df[1] == 'discrete', 0].tolist()
    continuous_data = X[:, db_enc.discrete_flen:]  # 连续特征部分
    # 将离散特征转换为词向量
    discrete_data_glove = transform_discrete_to_glove(X[:, :db_enc.discrete_flen], discrete_columns, glove_dict)

    # 将连续特征和离散特征的词向量合并
    combined_data = np.hstack([discrete_data_glove, continuous_data])

    # 找到每个样本的最近 K 个邻居
    k = 5  # 你想找的邻居数
    distances, indices = knn_find_k_nearest(combined_data, k)

    print("最近的邻居索引：", indices)
    print("最近的样本：", X[indices])
    print("对应的距离：", distances)

if __name__ == "__main__":
    main()
