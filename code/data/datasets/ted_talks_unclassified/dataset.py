import os
import pickle

current_folder = os.path.dirname(os.path.abspath(__file__))
FILENAME = os.path.join(current_folder, 'src', 'whitelist.npy')

def fetch(filename=FILENAME):
    with open(filename, 'rb') as f:
        X, Y = pickle.load(f)
    return X, Y
