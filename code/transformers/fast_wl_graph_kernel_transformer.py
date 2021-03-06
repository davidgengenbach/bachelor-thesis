import sklearn
import sklearn.preprocessing
from sklearn.preprocessing import normalize

from kernels import fast_wl
from utils import graph_helper
import numpy as np
import networkx as nx


def iteration_weight_function(iteration: int):
    return iteration + 2


def iteration_weight_constant(iteration: int, constant: int = 1):
    return constant


class FastWLGraphKernelTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(
            self,
            h=1,
            phi_dim=None,
            round_to_decimals=10,
            ignore_label_order=False,
            node_weight_function=None,
            node_weight_iteration_weight_function=iteration_weight_constant,
            use_early_stopping: bool = True,
            same_label: bool = False,
            use_directed: bool = True,
            truncate_to_highest_label: bool = True,
            norm: str = None,
            use_cached: bool = False,
            use_nx: bool = True
    ):
        self.h = h
        self.phi_dim = phi_dim
        self.round_to_decimals = round_to_decimals
        self.ignore_label_order = ignore_label_order
        self.node_weight_function = node_weight_function
        self.use_early_stopping = use_early_stopping
        self.node_weight_iteration_weight_function = node_weight_iteration_weight_function
        self.same_label = same_label
        self.use_directed = use_directed
        self.truncate_to_highest_label = truncate_to_highest_label
        self.norm = norm
        self.use_cached = use_cached
        self.use_nx = use_nx

    def fit(self, X, y=None, **fit_params):
        assert len(X)
        X, node_weight_factors = _retrieve_node_weights_and_convert_graphs(X, node_weight_function=self.node_weight_function, same_label=self.same_label, use_directed=self.use_directed, use_nx=self.use_nx)

        self.hashed_x = hash_dataset(X)
        phi_list, label_lookups, label_counters = fast_wl.transform(
            X,
            h=self.h,
            phi_dim=self.phi_dim,
            round_signatures_to_decimals=self.round_to_decimals,
            ignore_label_order=self.ignore_label_order,
            node_weight_factors=node_weight_factors,
            use_early_stopping=self.use_early_stopping,
            node_weight_iteration_weight_function=self.node_weight_iteration_weight_function,
            truncate_to_highest_label=self.truncate_to_highest_label
        )

        phi_list = _normalize(phi_list, norm=self.norm)

        self.phi_list_ = phi_list
        self.label_lookups_ = label_lookups
        self.label_counters_ = label_counters
        return self

    def transform(self, X, y=None, **fit_params):
        assert len(X)
        X, node_weight_factors = _retrieve_node_weights_and_convert_graphs(X, node_weight_function=self.node_weight_function, same_label=self.same_label, use_directed=self.use_directed, use_nx=self.use_nx)

        # Use already computed phi_list if the given X is the same as in fit()
        if self.use_cached and self.hashed_x == hash_dataset(X):
            return self.phi_list_

        # Use early stopping
        h = min(len(self.phi_list_) - 1, self.h)

        phi_dim = self.phi_dim
        if self.phi_dim is None:
            phi_dim = [phi.shape[1] for phi in self.phi_list_]

        phi_list, label_lookups, label_counters = fast_wl.transform(
            X,
            h=h,
            phi_dim=phi_dim,
            round_signatures_to_decimals=self.round_to_decimals,
            ignore_label_order=self.ignore_label_order,
            node_weight_factors=node_weight_factors,
            use_early_stopping=False,
            # Also give the label lookups/counters from training
            label_lookups=[dict(l) for l in self.label_lookups_],
            label_counters=self.label_counters_[:],
            node_weight_iteration_weight_function=self.node_weight_iteration_weight_function,
            truncate_to_highest_label=self.truncate_to_highest_label
        )

        phi_list = _normalize(phi_list, norm=self.norm)

        # phi_list = [x.T for x in phi_list]

        # Do NOT save label lookups and counters! This would effectively be fitting!
        return phi_list


def _normalize(phi_list, copy=False, norm=None):
    if norm:
        phi_list = [normalize(phi, norm=norm, copy=copy, axis=1) for phi in phi_list]
    return phi_list


def hash_dataset(X):
    return ''.join([str(hash(''.join([str(a) for a in labels]))) for adj, labels in X]) + str(len(X)) + str(X[int(len(X) / 2)][0].sum())


def get_node_weight_factors(X, metric=None):
    if metric is None:
        return None

    out = [metric(x) for x in X]
    if isinstance(out[0], dict):
        out = [[int(val) for key, val in sorted(val.items(), key=lambda x: x[0])] if len(val.keys()) else [1] for val in out]
    return out


def _retrieve_node_weights_and_convert_graphs(X, node_weight_function=None, same_label=False, use_directed=True, use_nx=True):
    X = graph_helper.get_graphs_only(X)
    if not use_directed:
        if use_nx:
            X = [nx.Graph(x) for x in X]
            assert not np.any([x.is_directed() for x in X])
        else:
            raise NotImplementedError('!use_directed and !use_nx not implemented')
    node_weight_factors = get_node_weight_factors(X, metric=node_weight_function)
    X = graph_helper.convert_graphs_to_adjs_tuples(X, copy=True)
    if same_label:
        X = [(adj, ['dummy'] * len(labels)) for adj, labels in X]

    return X, node_weight_factors

def iteration_weight_function_exponential(*a):
    raise NotImplementedError('Exponential function not implemented.')