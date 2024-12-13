import json
import math
import os
import sys
import time
import numpy as np
import cupy as cp
import pandas as pd
import concurrent.futures
from sklearn import preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, log_loss, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor
from tqdm import tqdm
from args import xgb_args
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rrl-DM_HW'))
from rrl.utils import read_csv

DATA_DIR = '../rrl-DM_HW/dataset'
EARLY_STOP = 50
CLASS_NUM = 2

class DBEncoder:
    """Encoder used for data discretization and binarization."""

    def __init__(self, f_df, drop='first', y_discrete=True):
        self.f_df = f_df
        self.y_discrete = y_discrete
        if y_discrete:
            self.label_enc = preprocessing.OneHotEncoder(categories='auto', handle_unknown='ignore')
        # Only initialize OneHotEncoder if not using raw discrete features
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
        if self.y_discrete:
            self.label_enc.fit(y_df)
        self.y_fname = y_df.columns

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if do not discretize them.
            self.imp.fit(continuous_data.values)
            self.mean = continuous_data.mean()
            self.std = continuous_data.std()
        
        if not discrete_data.empty:
            # Fit One-Hot Encoder if discrete=False
            self.feature_enc.fit(discrete_data)
        self.X_fname = list(discrete_data.columns) if not discrete_data.empty else []
        self.discrete_flen = discrete_data.shape[1]

        if continuous_data.shape[1] > 0:
            self.X_fname.extend(continuous_data.columns)
        self.continuous_flen = continuous_data.shape[1]
        
        self.f_df = pd.concat([self.f_df.loc[self.f_df[1] == 'discrete'], self.f_df.loc[self.f_df[1] == 'continuous']])

    def transform(self, X_df, y_df, normalized=False, keep_stat=False):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        y = y_df.values.flatten()

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if we do not discretize them.
            continuous_data = pd.DataFrame(self.imp.transform(continuous_data.values),
                                           columns=continuous_data.columns)
            if normalized:
                if keep_stat:
                    continuous_data = (continuous_data - continuous_data.mean()) / continuous_data.std()
                else:
                    continuous_data = (continuous_data - self.mean) / self.std

        if not discrete_data.empty:
            # Directly use discrete values as category features if discrete=True
            X_df = pd.concat([discrete_data.reset_index(drop=True), continuous_data.reset_index(drop=True)], axis=1)
        else:
            X_df = continuous_data

        return X_df.values, y

class XGBClassifierWithEarlyStopping(XGBClassifier):
    def fit(self, X, y, **kwargs):
        # Split X and y into train and eval set
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.05)
        super().fit(X_train, y_train, eval_set=[(X_eval, y_eval)])
        return self
        
class XGBRegressorWithEarlyStopping(XGBRegressor):
    def fit(self, X, y, **kwargs):
        # Split X and y into train and eval set
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.05)
        super().fit(X_train, y_train, eval_set=[(X_eval, y_eval)])
        return self

def f1_then_logloss(y_true, y_pred):
    # rationale: https://github.com/dmlc/xgboost/issues/10095
    y_true = y_true.reshape(-1, CLASS_NUM)
    y_true_class = np.argmax(y_true, axis=1)
    y_pred_class = np.argmax(y_pred, axis=1)
    # 计算logloss
    logloss_value = log_loss(y_true, y_pred)
    # 计算f1 macro score
    f1_macro = f1_score(y_true_class, y_pred_class, average='macro')

    ret = logloss_value + math.floor((1 - f1_macro) * 100)
    return ret

def get_data_loader(dataset, y_discrete=True):
    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)

    db_enc = DBEncoder(f_df, y_discrete=y_discrete)
    db_enc.fit(X_df, y_df)

    X, y = db_enc.transform(X_df, y_df)

    return X, y, db_enc

global_rule_list = []
def train_model(args):
    # Get data
    if args.task == 'classification-test':
        X_train, y_train, db_enc = get_data_loader(args.data_set.format('train'))
        X_test_df, y_test_df, _, _ = read_csv(os.path.join(DATA_DIR, args.data_set.format('test') + '.data'), os.path.join(DATA_DIR, args.data_set.format('test') + '.info'))
        X_test, y_test = db_enc.transform(X_test_df, y_test_df)
        y_discrete = True
    elif args.task == 'classification':
        X, y, db_enc = get_data_loader(args.data_set)
        y_discrete = True
    elif args.task == 'regression':
        X, y, db_enc = get_data_loader(args.data_set, y_discrete=False)
        y_discrete = False
    else:
        raise ValueError("Invalid task type. Please choose from 'classification', 'classification-test' or 'regression'.")

    # Define parameter grid
    # param_grid = {
    #     'k': args.knn, # [1, 3, 5]
    #     'n_jobs': [args.nthread],
    #     'batch_size': args.batch_size,
    #     'glove_file': ["glove.6B.50d.txt"]
    # }

    class XGB_DBEncoder:
        """Encoder used for data discretization and binarization."""

        def __init__(self, f_df, discrete_flen, continuous_flen, drop='first', y_discrete=True):
            self.f_df = f_df
            self.y_discrete = y_discrete
            if y_discrete:
                self.label_enc = db_enc.label_enc
            # Only initialize OneHotEncoder if not using raw discrete features
            self.feature_enc = db_enc.feature_enc
            self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            self.discrete_flen = discrete_flen
            self.continuous_flen = continuous_flen

        def split_data(self, X_inp: np.ndarray):
            discrete_data = X_inp[:, :self.discrete_flen]
            continuous_data = X_inp[:, self.discrete_flen:(self.discrete_flen + self.continuous_flen)]
            return discrete_data, continuous_data

        def fit(self, X_inp: np.ndarray, y_inp: np.ndarray):
            discrete_data, continuous_data = self.split_data(X_inp)
            y_df = pd.DataFrame(y_inp, columns=db_enc.y_fname)
            if self.y_discrete:
                self.y_fname = list(self.label_enc.get_feature_names_out(y_df.columns))
            else:
                self.y_fname = y_df.columns

            if self.continuous_flen > 0:
                # Use mean as missing value for continuous columns if do not discretize them.
                self.imp.fit(continuous_data)
            
        def transform(self, X_inp: np.ndarray, y_inp: np.ndarray):
            discrete_data, continuous_data = self.split_data(X_inp)
            # Encode string value to int index.
            if self.y_discrete:
                y = self.label_enc.transform(y_inp.reshape(-1, 1))
                y = y.toarray()
            else:
                y = y_inp

            if self.continuous_flen > 0:
                # Use mean as missing value for continuous columns if we do not discretize them.
                continuous_data = pd.DataFrame(self.imp.transform(continuous_data))
                
            if self.discrete_flen > 0:
                # One-hot encoding if discrete=False
                discrete_data = self.feature_enc.transform(discrete_data)
                X_df = pd.concat([pd.DataFrame(discrete_data.toarray()), continuous_data], axis=1)
            else:
                X_df = continuous_data

            return X_df.values, y

    class KNNEstimator(BaseEstimator, ClassifierMixin):
        def __init__(self, k=5, n_jobs=16, batch_size=3, glove_file="glove.6B.50d.txt"):
            self.k = k
            self.n_jobs = n_jobs
            self.glove_file = glove_file
            self.load_glove_embeddings(glove_file)
            self.discrete_columns = db_enc.f_df.loc[db_enc.f_df[1] == 'discrete', 0].tolist()
            self.continuous_columns = db_enc.f_df.loc[db_enc.f_df[1] == 'continuous', 0].tolist()
            self.glove_embed_size = 50
            self.batch_size = batch_size
            self.dis_dim = 2
            self.interpretable = False
            
        def load_glove_embeddings(self, glove_file):
            self.glove_dict = {}
            with open(glove_file, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.split()
                    word = values[0]
                    vector = np.asarray(values[1:], dtype='float32')
                    self.glove_dict[word] = vector

        def transform_discrete_to_glove(self, X_inp):
            glove_embeddings = []
            for cid, col in enumerate(self.discrete_columns):
                embeddings = []
                for value in X_inp[:, cid]:
                    if value.lower() in self.glove_dict:
                        embeddings.append(self.glove_dict[value.lower()][:min(self.glove_embed_size, self.dis_dim)])
                    else:
                        embeddings.append(np.zeros(min(self.glove_embed_size, self.dis_dim)))  # 如果词不在glove中，使用全0向量
                glove_embeddings.append(np.array(embeddings))
            return np.hstack(glove_embeddings) if len(glove_embeddings) > 0 else None

        def fit(self, X, y):
            self.X_ = X
            discrete_data_glove = self.transform_discrete_to_glove(X)
            self.nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='auto').fit(
                np.hstack([discrete_data_glove, X[:, db_enc.discrete_flen:]]) if discrete_data_glove is not None else X
            )
            self.y_ = y
            if args.task in ['classification', 'classification-test']:
                self.labels = np.unique(y)
                self.mode_label = pd.DataFrame(self.y_).mode().iloc[0][0]
            else:
                self.mode_label = self.y_.mean()
            return self

        def predict(self, X):
            samples = [X[i:i+1] for i in range(len(X))]
            result_list = []
            rule_list = []
            
            def majority_vote(labels):
                unique, counts = np.unique(labels, return_counts=True)
                return unique[np.argmax(counts)]
            
            def process_sample(sample):
                discrete_data_glove = self.transform_discrete_to_glove(sample)
                distances, indices = self.nbrs.kneighbors(
                    np.hstack([discrete_data_glove, sample[:, db_enc.discrete_flen:]]) if discrete_data_glove is not None else sample
                )
                
                indices = indices.flatten()
                indices = np.unique(indices)
                ref_samples = self.X_[indices]
                ref_labels = self.y_[indices]
                
                # print(ref_samples, ref_labels)
                xgb_db_enc = XGB_DBEncoder(db_enc.f_df, len(self.discrete_columns),  len(self.continuous_columns), y_discrete=y_discrete)
                xgb_db_enc.fit(ref_samples, ref_labels)
                train_samples, train_labels = xgb_db_enc.transform(ref_samples, ref_labels)
                test_samples, _ = xgb_db_enc.transform(sample, ref_labels[:1])
                
                # print(train_samples, train_labels)
                # print(test_samples)
                # print("================================================\n")
                # exit(0)
                if args.task in ['classification', 'classification-test']:
                    xgb_estimator = XGBClassifierWithEarlyStopping(
                        learning_rate=args.learning_rate[0],
                        gamma=args.gamma[0],
                        max_depth=args.max_depth[0],
                        eval_metric=f1_then_logloss,
                        early_stopping_rounds=EARLY_STOP,
                        n_jobs=1,
                        device=args.device
                    )
                    train_samples = cp.array(train_samples)
                    train_labels = cp.array(train_labels)
                    xgb_estimator.fit(train_samples, train_labels)
                    test_samples = cp.array(test_samples)
                    res = xgb_estimator.predict(test_samples)
                    res = xgb_db_enc.label_enc.inverse_transform(res)
                    res = [res_i[0] for res_i in res]

                    # if there is None in res, substitute None with voted training label
                    # print(res)
                    if None in res:
                        mv_label = majority_vote(ref_labels)
                        res = [mv_label if r is None else r for r in res]
                    # print(res)
                    # print("=============================\n")
                    
                elif args.task == 'regression':
                    xgb_estimator = XGBRegressorWithEarlyStopping(
                        learning_rate=args.learning_rate[0],
                        gamma=args.gamma[0],
                        max_depth=args.max_depth[0],
                        eval_metric='rmse',
                        early_stopping_rounds=EARLY_STOP,
                        n_jobs=1,
                        device=args.device
                    )
                    train_samples = cp.array(train_samples)
                    train_labels = cp.array(train_labels)
                    xgb_estimator.fit(train_samples, train_labels)
                    test_samples = cp.array(test_samples)
                    res = xgb_estimator.predict(test_samples)
                    # print(res)
                
                return res, []

            # 使用ThreadPoolExecutor进行多线程处理
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = [executor.submit(process_sample, sample) for sample in tqdm(samples, desc="Processing batches")]
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Completing futures"):
                    result, rules = future.result()
                    result_list.extend(result)
                    rule_list.extend(rules)
            
            global global_rule_list
            global_rule_list.append(rule_list)
            
            return np.array(result_list)


        def score(self, X, y):
            y_pred = self.predict(X)

            if args.task in ['classification', 'classification-test']:
                le = preprocessing.LabelEncoder()
                # 将字符串标签转换为数值
                le.fit(self.labels)
                y_encoded = le.transform(y)
                y_pred_encoded = le.transform(y_pred)
                res = f1_score(y_encoded, y_pred_encoded, average='macro')
            elif args.task == 'regression':
                res = mean_squared_error(y, y_pred, squared=False)
            return res

    # Create scoring metrics
    if args.task == 'classification':
        scoring = {
            'accuracy': make_scorer(accuracy_score),
            'precision_macro': make_scorer(precision_score, average='macro'),
            'recall_macro': make_scorer(recall_score, average='macro'),
            'f1_macro': make_scorer(f1_score, average='macro')
        }
    elif args.task == 'regression':
        scoring = {
            'rooted_mean_squared_error': make_scorer(mean_squared_error, squared=False)
        }

    # # Create GridSearchCV
    # grid_search = GridSearchCV(
    #     estimator=KNNEstimator(),
    #     param_grid=param_grid,
    #     scoring='f1_macro',
    #     cv=5,
    #     refit=True,
    #     return_train_score=True
    # )

    # # Fit the grid search
    # grid_search.fit(X, y)

    # # Get the best estimator
    # best_estimator = grid_search.best_estimator_
    # best_params = grid_search.best_params_

    # # Evaluate on cross-validation
    # cv_results = grid_search.cv_results_
    # mean_test_score = grid_search.best_score_

    # # Output the results
    # print("Best parameters found: ", best_params)
    # print("Best macro F1 score: ", mean_test_score)

    # Evaluate on the test set (or use cross-validation)
    # Since we used cross-validation, we can get the metrics from cv_results_

    best_estimator = KNNEstimator(k=args.knn[0], n_jobs=args.nthread, batch_size=args.batch_size[0], glove_file="../KNN_FS_LLM_code/glove.6B.50d.txt")

    if args.task == 'classification-test':
        fit_time = time.time()
        best_estimator.fit(X_train, y_train)
        fit_time = time.time() - fit_time
        score_time = time.time()
        y_pred = best_estimator.predict(X_test)
        score_time = time.time() - score_time
        
        le = preprocessing.LabelEncoder()
        # 将字符串标签转换为数值
        le.fit(best_estimator.labels)
        y_test_encoded = le.transform(y_test)
        y_pred_encoded = le.transform(y_pred)
        print("Test set results:")
        print("Accuracy: ", accuracy_score(y_test_encoded, y_pred_encoded))
        print("Macro Precision: ", precision_score(y_test_encoded, y_pred_encoded, average='macro'))
        print("Macro Recall: ", recall_score(y_test_encoded, y_pred_encoded, average='macro'))
        print("Macro F1 Score: ", f1_score(y_test_encoded, y_pred_encoded, average='macro'))
    
    else:
        # Let's get the scores for the best estimator
        metrics = cross_validate(best_estimator, X, y, cv=5, scoring=scoring)

        # Compute the mean of the metrics
        if args.task == 'classification':
            accuracy = np.mean(metrics['test_accuracy'])
            precision = np.mean(metrics['test_precision_macro'])
            recall = np.mean(metrics['test_recall_macro'])
            f1 = np.mean(metrics['test_f1_macro'])

            print("Accuracy: ", accuracy)
            print("Macro Precision: ", precision)
            print("Macro Recall: ", recall)
            print("Macro F1 Score: ", f1)
        elif args.task == 'regression':
            rmse = np.mean(metrics['test_rooted_mean_squared_error'])
            print("Rooted Mean Squared Error: ", rmse)
            
        fit_time = np.mean(metrics['fit_time'])
        score_time = np.mean(metrics['score_time'])

    print("Average fit time: ", fit_time)
    print("Average score time: ", score_time)
        
    print("Rule list:")
    print(json.dumps(global_rule_list, indent=4))

if __name__ == '__main__':
    train_model(xgb_args)
