import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--data_set', type=str, default='tic-tac-toe',
                    help='Set the data set for training. All the data sets in the dataset folder are available.')
parser.add_argument('-e', '--epoch', type=int, default=100, help='Set the number of boosting rounds (epochs).')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.1, help='Set the learning rate for XGBoost.')
parser.add_argument('-md', '--max_depth', type=int, default=6, help='Set the max depth of the trees.')
parser.add_argument('-g', '--gamma', type=float, default=0, help='Set the minimum loss reduction required to make a further partition.')
parser.add_argument('-nthread', '--nthread', type=int, default=2, help='Set the number of parallel threads.')
parser.add_argument('-ki', '--ith_kfold', type=int, default=0, help='Do the i-th 5-fold validation, 0 <= ki < 5.')
parser.add_argument('--print_feature_importance', action="store_true", help='Print feature importances after training.')

# Set paths for saving model and logs
parser.add_argument('--model', type=str, default='xgb_model.json', help='Path to save/load the model.')

xgb_args = parser.parse_args()
