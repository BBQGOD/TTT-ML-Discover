import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--data_set', type=str, default='tic-tac-toe',
                    help='Set the data set for training. All the data sets in the dataset folder are available.')
parser.add_argument('-t', '--task', type=str, default='classification',
                    help='Set the task for training. classification or regression.')
parser.add_argument('-ak', '--api_key', type=str, default='',
                    help='Set the API key for the LLM service.')
parser.add_argument('-bu', '--base_url', type=str, default='',
                    help='Set the base URL for the LLM service.')
parser.add_argument('-k', '--knn', type=int, nargs='+', default=[5], 
                    help='Set the number of neighbors for the KNN algorithm.')
parser.add_argument('-bs', '--batch_size', type=int, nargs='+', default=[3],
                    help='Set the batch size for training.')
parser.add_argument('-nthread', '--nthread', type=int, default=2, 
                    help='Set the number of parallel threads.')

llm_args = parser.parse_args()
