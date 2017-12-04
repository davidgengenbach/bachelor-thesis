import os
import pickle

current_folder = os.path.dirname(os.path.abspath(__file__))
FILENAME = os.path.join(current_folder, 'src', 'df_dataset.npy')

def fetch(filename=FILENAME):
    df = get_df(filename)
    X = df.transcript.values
    Y = df.label.values
    return X.tolist(), Y.tolist()

def get_df(filename=FILENAME):
    with open(filename, 'rb') as f:
        return pickle.load(f)