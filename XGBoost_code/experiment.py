import os
import sys
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, classification_report
from args import xgb_args
import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rrl-DM_HW'))
from rrl.utils import read_csv

DATA_DIR = '../rrl-DM_HW/dataset'


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
            continuous_data = continuous_data.astype(np.float)
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

    db_enc = DBEncoder(f_df, discrete=False)
    db_enc.fit(X_df, y_df)

    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)

    return X, y, db_enc

def train_model(args):
    # Get data
    X, y, db_enc = get_data_loader(args.dataset)

    # Define parameter grid
    param_grid = {
        'learning_rate': [0.01, 0.1, 0.2],
        'gamma': [0, 0.1, 1],
        'max_depth': [3, 5, 7]
    }

    # Create custom estimator with early stopping
    class XGBClassifierWithEarlyStopping(XGBClassifier):
        def fit(self, X, y, **kwargs):
            # Split X and y into train and eval set
            X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
            super().fit(X_train, y_train, eval_set=[(X_eval, y_eval)], eval_metric=['logloss', 'f1_macro'], early_stopping_rounds=10)
            return self

    # Create scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision_macro': make_scorer(precision_score, average='macro'),
        'recall_macro': make_scorer(recall_score, average='macro'),
        'f1_macro': make_scorer(f1_score, average='macro')
    }

    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=XGBClassifierWithEarlyStopping(),
        param_grid=param_grid,
        scoring='f1_macro',
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
    cv_results = grid_search.cv_results_
    mean_test_score = grid_search.best_score_

    # Output the results
    print("Best parameters found: ", best_params)
    print("Best macro F1 score: ", mean_test_score)

    # Evaluate on the test set (or use cross-validation)
    # Since we used cross-validation, we can get the metrics from cv_results_

    # Let's get the scores for the best estimator
    metrics = cross_validate(best_estimator, X, y, cv=5, scoring=scoring)

    # Compute the mean of the metrics
    accuracy = np.mean(metrics['test_accuracy'])
    precision = np.mean(metrics['test_precision_macro'])
    recall = np.mean(metrics['test_recall_macro'])
    f1 = np.mean(metrics['test_f1_macro'])

    print("Accuracy: ", accuracy)
    print("Macro Precision: ", precision)
    print("Macro Recall: ", recall)
    print("Macro F1 Score: ", f1)

    # Save the best model
    joblib.dump(best_estimator, 'best_xgb_model.pkl')

if __name__ == '__main__':
    train_model(xgb_args)
