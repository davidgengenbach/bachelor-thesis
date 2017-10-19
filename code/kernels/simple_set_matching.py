"""Calculates the jaccard coefficient of the node labels: jaccard_index = num_matching / num_all_labels

Similar to the "zero" iteration of WL? The phi feature maps per dot product and then normed by the total number of non-zero entries in phi?
"""
import numpy as np
import networkx as nx
import collections
import typing


def transform(graphs: typing.List, use_edge_labels: bool=False):
    K = np.eye(len(graphs))

    for i, graph_1 in enumerate(graphs):
        for j, graph_2 in enumerate(graphs[i + 1:]):
            if use_edge_labels:
                value = simple_set_matching_kernel_for_nx_graphs(graph_1, graph_2, use_edge_labels = True)
            else:
                value = simple_set_matching_kernel(graph_1, graph_2)
            K[i, i + j + 1] = value

    # Make gram matrix symmetric
    K = np.maximum(K, K.T)
    return K


def get_edge_names(g: nx.Graph):
    return [data['name'] for _, _, data in g.edges(data=True) if 'name' in data]


def simple_set_matching_kernel_for_nx_graphs(g1: nx.Graph, g2: nx.Graph, use_edge_labels: bool=False):
    nodes = [set(g1.nodes()), set(g2.nodes())]

    if use_edge_labels:
        nodes[0] |= set(get_edge_names(g1))
        nodes[1] |= set(get_edge_names(g2))

    return simple_set_matching_kernel(*nodes)


def simple_set_matching_kernel(labels_1: typing.List, labels_2: typing.List):
    labels_1, labels_2 = set(labels_1), set(labels_2)
    num_labels = len(labels_1 | labels_2)
    
    if num_labels == 0:
        return 0

    num_matching = len(labels_1 & labels_2)
    return num_matching / num_labels
