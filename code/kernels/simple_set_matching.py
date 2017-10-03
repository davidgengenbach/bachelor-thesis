"""Calculates the jaccard coefficient of the node labels: jaccard_index = num_matching / num_all_labels

Similar to the "zero" iteration of WL? The phi feature maps per dot product and then normed by the total number of non-zero entries in phi?
"""
import numpy as np


def transform(graphs):
    K = np.eye(len(graphs))

    for i, graph_1 in enumerate(graphs):
        for j, graph_2 in enumerate(graphs[i + 1:]):
            value = simple_set_matching_kernel(graph_1, graph_2)
            K[i, i + j + 1] = value

    # Make gram matrix symmetric
    K = np.maximum(K, K.T)
    return K


def simple_set_matching_kernel_for_nx_graphs(g1, g2):
    return simple_set_matching_kernel(g1.nodes(), g2.nodes())


def simple_set_matching_kernel(labels_1, labels_2):
    labels_1, labels_2 = set(labels_1), set(labels_2)
    num_labels = len(labels_1 | labels_2)
    
    if num_labels == 0:
        return 0

    num_matching = len(labels_1 & labels_2)
    return num_matching / num_labels
