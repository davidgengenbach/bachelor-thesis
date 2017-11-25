import os
import pickle
import re

_current_folder = os.path.dirname(os.path.abspath(__file__))
_dataset_file = os.path.join(_current_folder, 'src', 'dataset_imsdb.npy')

def fetch(remove_instructions=True):
    with open(_dataset_file, 'rb') as f:
        X, Y = pickle.load(f)

    if remove_instructions:
        for idx, x in enumerate(X):
            X[idx] = re.sub(r'(\(INSTRUCTION: .*?\))\.', '', x)
    return X, Y