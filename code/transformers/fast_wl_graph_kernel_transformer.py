import sklearn
from sklearn import base
import logging
import networkx as nx
import graph_helper
import wl
import fast_wl
from joblib import Parallel, delayed

def add_bogus_labels_to_empty_graphs(graphs):
    empty_graph_counter = 0
    for x in graphs:
        if nx.number_of_nodes(x) < 1:
            TEST_LABEL = 'ajksdlkajslkj'
            x.add_node(TEST_LABEL)
            x.add_edge(TEST_LABEL, TEST_LABEL)
            empty_graph_counter += 1
    return empty_graph_counter

class FastWLGraphKernelTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Fast Weisfeiler-Lehman transformer.
    
    Attributes:
        all_nodes (frozenset): all node labels in the training set
        h (int): the iterations for WL
        label_counters (list(int)): the label counters for each iteration 
        label_lookups (list(dict)): the dictionaries per iteration that are used to lookup labels
        phi_list (list(lil_matrix)): the phis for the iterations
        phi_shape (tuple(int)): the shape of phi (number of nodes X number of graphs)
        remove_missing_labels (boolean): whether to remove missing labels in the transform stage
        train_graph_count (int): how many graphs have been seen in the fit stage
    """

    def __init__(self, h=1, remove_missing_labels=True):
        self.remove_missing_labels = remove_missing_labels
        self.h = h

    def fit(self, X, y=None, **fit_params):
        """Initializes the list of node labels.
        
        Args:
            X (list(networkx.Graph)): the networkx graphs
            y (list(str), optional): the labels, not needed
        
        Returns:
            FastWLGraphKernelTransformer: returns self
        """
        print('FastWLGraphKernelTransformer.fit: len(X)={}, H={}'.format(len(X), self.h))

        empty_graph_counter = add_bogus_labels_to_empty_graphs(X)

        print("FastWLGraphKernelTransformer.fit: Found empty graphs in training set: {}".format(empty_graph_counter))
        
        phi_list, label_lookups, label_counters = fast_wl.fast_wl_compute(X, h=self.h)

        self.train_graph_count = len(X)
        self.all_nodes = graph_helper.get_all_node_labels(X, as_sorted_list = False)
        self.phi_list = phi_list
        self.phi_shape = self.phi_list[-1].shape
        self.label_lookups = label_lookups
        self.label_counters = label_counters
        return self

    def transform(self, X, y=None, **fit_params):
        print('FastWLGraphKernelTransformer.transform: len(X)={}, H={}'.format(len(X), self.h))

        # remove missing nodes
        if self.remove_missing_labels:
            for graph in X:
                missing_nodes = frozenset(graph.nodes()) - self.all_nodes
                if len(missing_nodes):
                    graph.remove_nodes_from(missing_nodes)
        empty_graph_counter = add_bogus_labels_to_empty_graphs(X)

        phi_list, label_lookups, label_counters = fast_wl.fast_wl_compute(X, h=self.h, label_lookups=self.label_lookups, label_counters=self.label_counters, phi_dim = self.phi_shape[0])

        self.label_lookups = label_lookups
        self.label_counters = label_counters
        return phi_list