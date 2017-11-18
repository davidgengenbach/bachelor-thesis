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
