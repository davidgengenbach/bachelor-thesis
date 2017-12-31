import numpy as np
import networkx as nx

def adj_node_degree_metric(x):
    assert isinstance(x, tuple)
    adj, labels = x
    return adj.sum(axis=1, dtype=np.uint32)


def nxgraph_pagerank_metric(x):
    return {k: v * 100 for k, v in nx.pagerank(nx.Graph(x)).items()}


def nxgraph_degrees_metric(x):
    return x.degree()


def adj_degrees_metric(x):
    assert isinstance(x, tuple)
    adj, labels = x
    return np.squeeze(np.asarray(adj.sum(axis=1)))


def get_node_weights_for_nxgraphs(X, metric=None, add_for_empty_graphs = True):
    if not metric: return None
    assert isinstance(X[0], nx.Graph)

    metrics = [metric(x) for x in X]
    assert np.all([isinstance(x, dict) for x in metrics])

    return [[metric_[l] for l in sorted(graph.nodes())] if len(graph.nodes()) else [1] if add_for_empty_graphs else [] for graph, metric_ in zip(X, metrics)]