import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--data_set', type=str, default='tic-tac-toe',
                    help='Set the data set for training. All the data sets in the dataset folder are available.')
parser.add_argument('-t', '--task', type=str, default='classification',
                    help='Set the task for training. classification or regression.')
parser.add_argument('-lr', '--learning_rate', type=float, nargs='+', default=[0.1], 
                    help='Set the learning rate(s) for XGBoost. Provide multiple values as a list.')
parser.add_argument('-md', '--max_depth', type=int, nargs='+', default=[3], 
                    help='Set the max depth(s) of the trees. Provide multiple values as a list.')
parser.add_argument('-g', '--gamma', type=float, nargs='+', default=[0], 
                    help='Set the gamma(s) (minimum loss reduction required to make a further partition). Provide multiple values as a list.')
parser.add_argument('-nthread', '--nthread', type=int, default=2, 
                    help='Set the number of parallel threads.')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'],
                    help='Set the device for training. cpu or cuda.')

parser.add_argument('-k', '--knn', type=int, nargs='+', default=[50], 
                    help='Set the number of neighbors for the KNN algorithm.')
parser.add_argument('-bs', '--batch_size', type=int, nargs='+', default=[3],
                    help='Set the batch size for training.')


# Set paths for saving model and logs
parser.add_argument('--model', type=str, default='xgb_model.json', help='Path to save/load the model.')

xgb_args = parser.parse_args()

# if xgb_args.device == 'cuda':
    # xgb_args.nthread = 1
