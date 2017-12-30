import sklearn
import collections
import networkx as nx
from utils import graph_helper
import numpy as np
from itertools import chain
import scipy.sparse


class RemoveNewGraphNodes(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, copy=True):
        self.train_labels = None
        self.copy = copy

    def fit(self, X, *s):
        self.train_labels = set(chain.from_iterable([_get_labels(x) for x in graph_helper.get_graphs_only(X)]))
        return self

    def transform(self, X, y=None, **fit_params):
        assert len(X)

        X = graph_helper.get_graphs_only(X)
        if self.copy:
            X = _copy_graphs(X)

        X = [_keep_only_train_labels(x, self.train_labels) for x in X]
        return X


def _get_labels(x):
    if isinstance(x, tuple):
        return x[1]
    elif isinstance(x, nx.Graph):
        return x.nodes()
    else:
        raise Exception('Invalid graph type: {}'.format(x))


def _copy_graphs(X):
    is_adj = isinstance(X[0], tuple)
    if is_adj:
        return [(adj.copy(), labels[:]) for adj, labels in X]
    else:
        return [g.copy() for g in X]


def _keep_only_train_labels(g, labels_to_keep):
    if isinstance(g, tuple):
        assert len(g) == 2
        adj, labels = g
        label_indices = []
        new_labels = labels[:]

        for idx, x in enumerate(labels):
            if x in labels_to_keep: continue
            label_indices.append(idx)
            new_labels.remove(x)

        if len(label_indices):
            keep_indices = [x for x in range(adj.shape[0]) if x not in label_indices]
            keep_indices = np.ix_(keep_indices, keep_indices)
            adj = adj[keep_indices]
        return adj, new_labels
    elif isinstance(g, nx.Graph):
        nodes = set(g.nodes())
        labels_to_remove = nodes - labels_to_keep
        g.remove_nodes_from(labels_to_remove)
        return g
    else:
        raise Exception('Unknown graph type: {}'.format(g))
