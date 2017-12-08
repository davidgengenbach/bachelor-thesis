import numpy as np

from utils import dataset_helper
import collections


def fetch(dataset='nyt', num_per_class=20):
    X, Y = dataset_helper.get_dataset(dataset)

    per_class = collections.defaultdict(list)
    for x, y in zip(X, Y):
        per_class[y].append(x)

    assert np.all(len(x) >= num_per_class for x in per_class.values())

    X_, Y_ = [], []
    for y, xs in per_class.items():
        xs_ = np.random.choice(xs, size=num_per_class)
        Y_ += [y] * len(xs_)
        X_ += xs_.tolist()
    assert len(X_) == len(Y_)

    return X_, Y_
