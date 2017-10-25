import sklearn
from sklearn import base
from kernels import fast_wl
from utils import graph_helper
import numpy as np


def hash_dataset(X):
    return ''.join([str(hash(''.join([str(a) for a in labels]))) for adj, labels in X])

class FastWLGraphKernelTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Fast Weisfeiler-Lehman transformer.
    
    Attributes:
        all_nodes (frozenset): all node labels in the training set
        debug (bool): whether do log more
        h (int): the iterations for WL
        label_counters (list(int)): the label counters for each iteration 
        label_lookups (list(dict)): the dictionaries per iteration that are used to lookup labels
        phi_list (list(lil_matrix)): the phis for the iterations
        phi_shape (tuple(int)): the shape of phi (number of nodes X number of graphs)
        remove_missing_labels (boolean): whether to remove missing labels in the transform stage
        train_graph_count (int): how many graphs have been seen in the fit stage
    """

    def __init__(self, h=1, remove_missing_labels=True, debug=False, should_cast = False, phi_dim = None, round_to_decimals = 10, ignore_label_order = False):
        self.remove_missing_labels = remove_missing_labels
        self.h = h
        self.debug = debug
        self.should_cast = should_cast
        self.phi_dim = phi_dim
        self.round_to_decimals = round_to_decimals
        self.ignore_label_order = ignore_label_order

    def fit(self, X, y=None, **fit_params):
        """Initializes the list of node labels.

        Args:
            X (list(networkx.Graph)): the networkx graphs
            y (list(str), optional): the labels, not needed

        Returns:
            FastWLGraphKernelTransformer: returns self
        """
        assert len(X)
        graph_helper.convert_graphs_to_adjs_tuples(X)
        # Remove empty graphs
        X = [x for x in X if x is not None]
        self.hashed_x = hash_dataset(X)

        if self.debug:
            print('FastWLGraphKernelTransformer.fit: len(X)={}, H={}'.format(len(X), self.h))

        empty_graph_counter = graph_helper.get_empty_graphs_count(X)

        if self.debug:
            print("FastWLGraphKernelTransformer.fit: Found empty graphs in training set: {}".format(empty_graph_counter))

        phi_list, label_lookups, label_counters = fast_wl.transform(X, h=self.h, cast_after_rounding=self.should_cast, phi_dim=self.phi_dim, round_signatures_to_decimals=self.round_to_decimals, ignore_label_order = self.ignore_label_order)

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

        graph_helper.convert_graphs_to_adjs_tuples(X)

        # Use already computed phi_list if the given X is the same as in fit()
        if self.hashed_x == hash_dataset(X):
            return self.phi_list

        # remove missing nodes
        if self.remove_missing_labels:
            for adj, labels in X:
                missing_labels = set(labels)- self.all_nodes
                # TODO
                if len(missing_labels):
                    pass

        phi_list, label_lookups, label_counters = fast_wl.transform(
            X, h=self.h, label_lookups=np.copy(self.label_lookups), label_counters=np.copy(self.label_counters), phi_dim=self.phi_shape[1], cast_after_rounding=self.should_cast, append_to_labels = True,round_signatures_to_decimals=self.round_to_decimals, ignore_label_order = self.ignore_label_order)

        # Do NOT save label lookups and counters! This would effectively be fitting!
        #self.label_lookups = label_lookups
        #self.label_counters = label_counters
        return [x.T for x in phi_list]
