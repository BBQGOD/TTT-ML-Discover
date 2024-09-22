import numpy as np
from sklearn.neighbors import NearestNeighbors

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
def transform_discrete_to_glove(X_df, discrete_columns, glove_dict, embedding_size=50):
    glove_embeddings = []
    for col in discrete_columns:
        embeddings = []
        for value in X_df[col]:
            if value in glove_dict:
                embeddings.append(glove_dict[value])
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
    X, y, db_enc = get_data_loader('your_dataset')

    # 加载 GloVe 词向量
    glove_file = 'path_to_glove.txt'  # 例如 'glove.6B.50d.txt'
    glove_dict = load_glove_embeddings(glove_file)

    # 获取离散和连续特征
    discrete_columns = db_enc.f_df.loc[db_enc.f_df[1] == 'discrete', 0].tolist()
    continuous_data = X[:, db_enc.discrete_flen:]  # 连续特征部分

    # 将离散特征转换为词向量
    discrete_data_glove = transform_discrete_to_glove(pd.DataFrame(X[:, :db_enc.discrete_flen]), discrete_columns, glove_dict)

    # 将连续特征和离散特征的词向量合并
    combined_data = np.hstack([discrete_data_glove, continuous_data])

    # 找到每个样本的最近 K 个邻居
    k = 5  # 你想找的邻居数
    distances, indices = knn_find_k_nearest(combined_data, k)

    print("最近的邻居索引：", indices)
    print("对应的距离：", distances)

if __name__ == "__main__":
    main()
