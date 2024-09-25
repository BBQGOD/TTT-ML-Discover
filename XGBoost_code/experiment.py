import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import make_scorer, accuracy_score, mean_squared_error, precision_score, recall_score, f1_score, log_loss
from args import xgb_args
import math
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rrl-DM_HW'))
from rrl.utils import read_csv

DATA_DIR = '../rrl-DM_HW/dataset'
EARLY_STOP = 50
CLASS_NUM = 2


class DBEncoder:
    """Encoder used for data discretization and binarization."""

    def __init__(self, f_df, discrete=False, y_one_hot=True, drop='first', y_discrete=True):
        self.f_df = f_df
        self.discrete = discrete
        self.y_discrete = y_discrete
        self.y_one_hot = y_one_hot
        if y_discrete:
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
        if self.y_discrete:
            self.label_enc.fit(y_df)
            self.y_fname = list(self.label_enc.get_feature_names_out(y_df.columns)) if self.y_one_hot else y_df.columns
        else:
            self.y_fname = y_df.columns

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if do not discretize them.
            self.imp.fit(continuous_data.values)
            self.mean = continuous_data.mean()
            self.std = continuous_data.std()
        
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
        if self.y_discrete:
            y = self.label_enc.transform(y_df.values.reshape(-1, 1))
            if self.y_one_hot:
                y = y.toarray()
        else:
            y = y_df.values

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

def get_data_loader(dataset, y_discrete=True):
    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)

    db_enc = DBEncoder(f_df, discrete=False, y_discrete=y_discrete)
    db_enc.fit(X_df, y_df)

    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)

    return X, y, db_enc

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

def train_model(args):
    # Get data
    if args.task == 'classification-test':
        X_train, y_train, db_enc = get_data_loader(args.data_set.format('train'))
        X_test_df, y_test_df, _, _ = read_csv(os.path.join(DATA_DIR, args.data_set.format('test') + '.data'), os.path.join(DATA_DIR, args.data_set.format('test') + '.info'))
        X_test, y_test = db_enc.transform(X_test_df, y_test_df, normalized=True)
    elif args.task == 'classification':
        X, y, db_enc = get_data_loader(args.data_set)
    elif args.task == 'regression':
        X, y, db_enc = get_data_loader(args.data_set, y_discrete=False)
    else:
        raise ValueError("Invalid task type. Please choose from 'classification', 'classification-test' or 'regression'.")

    # Define parameter grid
    param_grid = {
        'learning_rate': args.learning_rate,
        'gamma': args.gamma,
        'max_depth': args.max_depth,
        'eval_metric': [f1_then_logloss if args.task == 'classification' else 'rmse'],
        'early_stopping_rounds': [EARLY_STOP],
        'n_jobs': [args.nthread]
    }

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
        
    if args.task == 'classification-test':
        best_estimator = XGBClassifierWithEarlyStopping(
            learning_rate=args.learning_rate[0],
            gamma=args.gamma[0],
            max_depth=args.max_depth[0],
            eval_metric=f1_then_logloss,
            early_stopping_rounds=EARLY_STOP,
            n_jobs=args.nthread
        )
        fit_time = time.time()
        best_estimator.fit(X_train, y_train)
        fit_time = time.time() - fit_time
        score_time = time.time()
        y_pred = best_estimator.predict(X_test)
        score_time = time.time() - score_time
        
        print("Test set results:")
        print("Accuracy: ", accuracy_score(y_test, y_pred))
        print("Macro Precision: ", precision_score(y_test, y_pred, average='macro'))
        print("Macro Recall: ", recall_score(y_test, y_pred, average='macro'))
        print("Macro F1 Score: ", f1_score(y_test, y_pred, average='macro'))
    
    else:

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

        # Create GridSearchCV
        if args.task == 'classification':
            grid_search = GridSearchCV(
                estimator=XGBClassifierWithEarlyStopping(),
                param_grid=param_grid,
                scoring='f1_macro',
                cv=5,
                refit=True,
                return_train_score=True
            )
        elif args.task == 'regression':
            grid_search = GridSearchCV(
                estimator=XGBRegressorWithEarlyStopping(),
                param_grid=param_grid,
                scoring='neg_root_mean_squared_error',
                cv=5,
                refit=True,
                return_train_score=True
            )

        # Fit the grid search
        grid_search.fit(X, y)

        # Get the best estimator
        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_

        # Evaluate on cross-validation
        # cv_results = grid_search.cv_results_
        # mean_test_score = grid_search.best_score_

        # Output the results
        print("Best parameters found: ", best_params)
        # print("Best macro F1 score: ", mean_test_score)

        # Evaluate on the test set (or use cross-validation)
        # Since we used cross-validation, we can get the metrics from cv_results_

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
        
    # Save the best model
    # best_estimator.save_model(args.model)

if __name__ == '__main__':
    train_model(xgb_args)
