import sklearn
from sklearn import base
import logging
import networkx as nx
import graph_helper
import wl
import fast_wl
from joblib import Parallel, delayed

class FastWLGraphKernelTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Weisfeiler-Lehman transformer. Fits the training set by calculating the phi for each instance

    Attributes:
        all_nodes (frozenset): all node labels in the training set
        label_counters (list(int)): the label counters for each iteration 
        label_lookups (list(dict)): the dictionaries per iteration that are used to lookup labels
        phi_list (list(lil_matrix)): the phis for the iterations
    """

    def __init__(self, H=1, remove_missing_labels=True, n_jobs = 1):
        self.remove_missing_labels = remove_missing_labels
        self.H = H
        self.n_jobs = n_jobs

    def fit(self, X, y=None, **fit_params):
        """Initializes the list of node labels.

        Args:
            X (list(networkx.Graph)): the networkx graphs
            y (list(str), optional): the labels, not needed

        Returns:
            FastWLGraphKernelTransformer: returns self
        """
        print('FastWLGraphKernelTransformer.fit: len(X)={}, H={}'.format(len(X), self.H))

        empty_graph_counter = 0
        for x in X:
            if nx.number_of_nodes(x) < 1:
                TEST_LABEL = 'ajksdlkajslkj'
                x.add_node(TEST_LABEL)
                x.add_edge(TEST_LABEL, TEST_LABEL)
                empty_graph_counter += 1
                
        print("FastWLGraphKernelTransformer.fit: Found empty graphs in training set: {}".format(empty_graph_counter))

        phi_list, label_lookups, label_counters = fast_wl.fast_wl_compute(X, h=self.H)

        self.phi_list = phi_list[-1]
        self.phi_shape = self.phi_list.shape
        self.label_lookups = label_lookups
        self.label_counters = label_counters
        self.train_graph_count = len(X)
        return self

    def transform(self, X, y=None, **fit_params):
        print('FastWLGraphKernelTransformer.transform: len(X)={}, H={}'.format(len(X), self.H))
        # This is to cache the prior transformation of the training samples.
        # This is deeply problematic and must be changed eventually!
        if len(X) == self.train_graph_count:
            print('FastWLGraphKernelTransformer.transform: using pre-calculated phi list')
            return self.phi_list.T

        # remove missing nodes
        if self.remove_missing_labels:
            for graph in X:
                missing_nodes = frozenset(graph.nodes()) - self.all_nodes
                if len(missing_nodes):
                    graph = graph.copy()
                    graph.remove_nodes_from(missing_nodes)

        phi_list, label_lookups, label_counters = fast_wl.fast_wl_compute(graphs, h=self.H, label_lookups=self.label_lookups, label_counters=self.label_counters, phi_dim = self.phi_shape[0], primes_arguments_required=primes_arguments_required, labels_dtype = np.uint32)

        self.label_lookups = label_lookups
        self.label_counters = label_counters

        return phi_list[-1].T
