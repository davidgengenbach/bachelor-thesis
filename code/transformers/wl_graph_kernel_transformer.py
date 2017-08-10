import sklearn
from sklearn import base
import logging
import networkx as nx
import graph_helper
import wl
from joblib import Parallel, delayed

class WLGraphKernelTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
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
            WLGraphKernelTransformer: returns self
        """
        print('WLGraphKernelTransformer.fit: len(X)={}, H={}'.format(len(X), self.H))
        all_nodes = set()
        for x in X:
            all_nodes |= set(x.nodes())
        self.all_nodes = frozenset(all_nodes)

        node_label = [sorted(g.nodes()) for g in X]

        empty_graph_counter = 0
        for idx, labels in enumerate(node_label):
            if len(labels) == 0:
                TEST_LABEL = 'ajksdlkajslkj'
                X[idx].add_node(TEST_LABEL)
                X[idx].add_edge(TEST_LABEL, TEST_LABEL)
                labels.append(TEST_LABEL)
                empty_graph_counter += 1
        print("WLGraphKernelTransformer.fit: Found empty graphs in training set: {}".format(empty_graph_counter))

        ad_list = [nx.adjacency_matrix(g, nodelist=labels) for g, labels in zip(X, node_label)]
        # TODO: do this in batches
        phi_list, label_lookups, label_counters = wl.WL_compute_batched(
            adjs = ad_list, node_label = node_label, h = self.H, all_nodes=self.all_nodes, compute_k=False, DEBUG=True)

        self.phi_list = phi_list[-1]
        self.phi_shape = self.phi_list.shape
        self.label_lookups = label_lookups
        self.label_counters = label_counters
        self.train_graph_count = len(X)
        del ad_list, node_label
        return self

    def transform(self, X, y=None, **fit_params):
        print('WLGraphKernelTransformer.transform: len(X)={}, H={}'.format(len(X), self.H))
        # This is to cache the prior transformation of the training samples.
        # This is deeply problematic and must be changed eventually!
        if len(X) == self.train_graph_count:
            print('WLGraphKernelTransformer.transform: using pre-calculated phi list')
            return self.phi_list.T


        # remove missing nodes
        if self.remove_missing_labels:
            for graph in X:
                missing_nodes = frozenset(graph.nodes()) - self.all_nodes
                if len(missing_nodes):
                    graph = graph.copy()
                    graph.remove_nodes_from(missing_nodes)

        adjs, nodes = graph_helper.get_wl_args(X)
        phi_list, current_label_lookups, current_label_counters = wl.WL_compute_batched(
            adjs=adjs,
            node_label=nodes,
            batch_size=1000,
            all_nodes = self.all_nodes,
            compute_k = False,
            initial_label_lookups = self.label_lookups,
            initial_label_counters = self.label_counters,
            phi_dim = self.phi_shape[0],
            h = self.H,
            DEBUG = True
        )
        # Execute kernel
        return phi_list[-1]
