import networkx as nx
import numpy as np
import sklearn

from kernels import fast_wl
from utils import graph_helper


def hash_dataset(X):
    return ''.join([str(hash(''.join([str(a) for a in labels]))) for adj, labels in X])

def pagerank_metric(x):
    return {k: v * 100 for k, v in nx.pagerank(nx.Graph(x)).items()}

def degrees_metric(x):
    return x.degree()
    #return [adj.sum(axis = 1, dtype=np.uint32) for adj, _ in X]

def get_node_weight_factors(X, metric = pagerank_metric, use_node_weight_factors = True):
    if not use_node_weight_factors:
        return None
    out = [metric(x) for x in X]
    out = [[int(val) for key, val in sorted(val.items(), key = lambda x: x[0])] for val in out]
    return out


class FastWLGraphKernelTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Fast Weisfeiler-Lehman transformer.
    
    Attributes:

    """

    def __init__(self, h=1, debug=False, should_cast = False, phi_dim = None, round_to_decimals = 10, ignore_label_order = False, use_node_weight_factors = False, node_weight_function = pagerank_metric):
        self.h = h
        self.debug = debug
        self.should_cast = should_cast
        self.phi_dim = phi_dim
        self.round_to_decimals = round_to_decimals
        self.ignore_label_order = ignore_label_order
        self.use_node_weight_factors = use_node_weight_factors
        self.node_weight_function = node_weight_function

    def fit(self, X, y=None, **fit_params):
        """Initializes the list of node labels.

        Args:
            X (list(networkx.Graph)): the networkx graphs
            y (list(str), optional): the labels, not needed

        Returns:
            FastWLGraphKernelTransformer: returns self
        """
        assert len(X)

        X = graph_helper.get_graphs_only(X)

        assert isinstance(X[0], nx.Graph)
        node_weight_factors = get_node_weight_factors(X, metric = self.node_weight_function, use_node_weight_factors=self.use_node_weight_factors)
        X = graph_helper.convert_graphs_to_adjs_tuples(X, copy = True)


        # Remove empty graphs
        X = [x for x in X if x is not None]
        self.hashed_x = hash_dataset(X)

        if self.debug:
            print('FastWLGraphKernelTransformer.fit: len(X)={}, H={}'.format(len(X), self.h))

        empty_graph_counter = graph_helper.get_empty_graphs_count(X)

        if self.debug:
            print("FastWLGraphKernelTransformer.fit: Found empty graphs in training set: {}".format(empty_graph_counter))

        phi_list, label_lookups, label_counters = fast_wl.transform(X, h=self.h, phi_dim=self.phi_dim, round_signatures_to_decimals=self.round_to_decimals, ignore_label_order = self.ignore_label_order, node_weight_factors=node_weight_factors)

        self.train_graph_count = len(X)
        self.all_nodes = graph_helper.get_all_node_labels(X, as_sorted_list=False)
        self.phi_list = [x.T for x in phi_list]
        self.phi_shape = self.phi_list[-1].shape
        self.label_lookups = label_lookups
        self.label_counters = label_counters

        return self

    def transform(self, X, y=None, **fit_params):
        if self.debug:
            print('FastWLGraphKernelTransformer.transform: len(X)={}, H={}'.format(len(X), self.h))

        X = graph_helper.get_graphs_only(X)
        node_weight_factors = get_node_weight_factors(X, metric = self.node_weight_function, use_node_weight_factors=self.use_node_weight_factors)
        X = graph_helper.convert_graphs_to_adjs_tuples(X, copy=True)

        # Use already computed phi_list if the given X is the same as in fit()
        if self.hashed_x == hash_dataset(X):
            return self.phi_list

        phi_list, label_lookups, label_counters = fast_wl.transform(
            X, h=self.h, label_lookups=np.copy(self.label_lookups), label_counters=np.copy(self.label_counters), phi_dim=self.phi_shape[1], append_to_labels = True, round_signatures_to_decimals=self.round_to_decimals, ignore_label_order = self.ignore_label_order, node_weight_factors=node_weight_factors)

        # Do NOT save label lookups and counters! This would effectively be fitting!
        #self.label_lookups = label_lookups
        #self.label_counters = label_counters
        return [x.T for x in phi_list]
