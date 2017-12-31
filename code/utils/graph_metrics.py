import numpy as np
import networkx as nx


def nxgraph_pagerank_metric(x):
    return {k: v * 100 for k, v in nx.pagerank(nx.Graph(x)).items()}


def nxgraph_degrees_metric(x):
    return x.degree()


def nxgraph_degrees_metric_max(x):
    degrees = x.degree()
    if not len(degrees):
        return degrees
    max_ = np.max([1] + list(degrees.values()))
    return {k: v / max_ for k, v in degrees.items()}


def adj_degrees_metric(x):
    assert isinstance(x, tuple)
    adj, labels = x
    return np.squeeze(np.asarray(adj.sum(axis=1)))


def adj_degrees_metric_max(x):
    factors = adj_degrees_metric(x)
    max_ = np.max(factors)

    if max_ > 1:
        factors = factors / np.max(factors)

    return np.where(factors < 0.5, 0.5, factors)
