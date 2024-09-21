import os
import sys
import functools
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from args import xgb_args
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rrl-DM_HW'))
from rrl.utils import read_csv, DBEncoder

DATA_DIR = '../rrl-DM_HW/dataset'


def get_data_loader(dataset, k=0):
    data_path = os.path.join(DATA_DIR, dataset + '.data')
    info_path = os.path.join(DATA_DIR, dataset + '.info')
    X_df, y_df, f_df, label_pos = read_csv(data_path, info_path, shuffle=True)

    db_enc = DBEncoder(f_df, discrete=False)
    db_enc.fit(X_df, y_df)

    X, y = db_enc.transform(X_df, y_df, normalized=True, keep_stat=True)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    train_index, test_index = list(kf.split(X_df))[k]
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]

    return X_train, y_train, X_test, y_test, db_enc

def train_model(args):
    dataset = args.data_set
    X_train, y_train, X_test, y_test, db_enc = get_data_loader(dataset, k=args.ith_kfold)

    macro_f1_score = functools.partial(f1_score, average='macro')

    # Initialize XGBoost model
    model = XGBClassifier(
        learning_rate=args.learning_rate,
        n_estimators=args.epoch,
        max_depth=args.max_depth,
        gamma=args.gamma,
        n_jobs=args.nthread,
        eval_metric=['logloss'],
        # eval_metric=['logloss', macro_f1_score],  # Simultaneously evaluate logloss and macro-f1
        early_stopping_rounds=10
    )

    # Train the model with early stopping and custom F1 evaluation
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=True)

    # Save the best model based on validation score
    best_iteration = model.best_iteration
    model.save_model(args.model)
    print(f"Best model saved to {args.model} at iteration {best_iteration}")

def test_model(args):
    dataset = args.data_set
    _, _, X_test, y_test, db_enc = get_data_loader(dataset, k=args.ith_kfold)

    # Load the model
    model = XGBClassifier()
    model.load_model(args.model)
    print(f"Model loaded from {args.model}")

    # Make predictions (返回的已经是类别标签)
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    recall_weighted = recall_score(y_test, y_pred, average='weighted')
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')

    # Print the results
    print(f"Accuracy: {accuracy * 100.0:.2f}%")
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Precision (weighted): {precision_weighted:.4f}")
    print(f"Recall (macro): {recall_macro:.4f}")
    print(f"Recall (weighted): {recall_weighted:.4f}")
    print(f"F1 Score (macro): {f1_macro:.4f}")
    print(f"F1 Score (weighted): {f1_weighted:.4f}")

    # Optionally, print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Optionally, print feature importance
    if args.print_feature_importance:
        print("Feature importances:")
        print(model.feature_importances_)

if __name__ == '__main__':
    train_model(xgb_args)
    test_model(xgb_args)
