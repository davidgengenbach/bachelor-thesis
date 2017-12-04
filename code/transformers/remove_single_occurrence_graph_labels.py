import sklearn
import collections
import networkx as nx
from utils import helper, graph_helper
import numpy as np
from itertools import chain
import scipy.sparse


class RemoveSingleOccurrenceGraphLabels(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, copy=True):
        self.copy = copy

    def fit(self, X, *s):
        return self

    def transform(self, X, y=None, **fit_params):
        assert len(X)
        X = graph_helper.get_graphs_only(X)
        if self.copy:
            X = _copy_graphs(X)
        labels = [_get_labels(x) for x in X]
        occurrences = collections.Counter(chain.from_iterable(labels))
        labels_to_be_removed = [k for k, v in occurrences.items() if v == 1]
        X = [_remove_label(x, labels_to_be_removed) for x in X]
        return X


def _copy_graphs(X):
    is_adj = isinstance(X[0], tuple)
    if is_adj:
        return [(adj.copy(), labels[:]) for adj, labels in X]
    else:
        return [g.copy() for g in X]


def _get_labels(x):
    if isinstance(x, tuple):
        return x[1]
    elif isinstance(x, nx.Graph):
        return x.nodes()
    else:
        raise Exception('Invalid graph type: {}'.format(x))


def _remove_label(g, labels_to_be_removed):
    if isinstance(g, tuple):
        assert len(g) == 2
        adj, labels = g

        label_indices = []
        for x in labels_to_be_removed:
            if x not in labels: continue
            label_indices.append(labels.index(x))
            labels.remove(x)
        if len(label_indices):
            keep_indices = [x for x in range(adj.shape[0]) if x not in label_indices]
            keep_indices = np.ix_(keep_indices, keep_indices)
            adj = adj[keep_indices]
        return adj, labels
    elif isinstance(g, nx.Graph):
        g.remove_nodes_from(labels_to_be_removed)
        return g
    else:
        raise Exception('Unknown graph type: {}'.format(g))
