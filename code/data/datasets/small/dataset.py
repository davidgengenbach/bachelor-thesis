import sklearn
import dataset_helper
import numpy as np

def fetch(dataset = 'ng20'):
    X, Y = dataset_helper.get_dataset(dataset)
    X, Y = np.array(X, dtype=object), np.array(Y, dtype=object)
    sss = sklearn.model_selection.StratifiedShuffleSplit(n_splits = 20, random_state=42)
    for train_index, test_index in sss.split(X, Y):
        X_test, Y_test = X[test_index], Y[test_index]
        return X_test.tolist(), Y_test.tolist()