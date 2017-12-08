import os
import pickle

current_folder = os.path.dirname(os.path.abspath(__file__))
FILENAME = os.path.join(current_folder, 'src', 'dataset_nyt.npy')

def fetch(filename=FILENAME):
    with open(filename, 'rb') as f:
        X, Y = pickle.load(f)
    X, Y = X.tolist(), Y.tolist()
    X = [x.replace("''", '') for x in X]
    return X, Y
