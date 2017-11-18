import sklearn

from kernels import fast_wl
from utils import graph_helper


class FastWLGraphKernelTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    def __init__(self, h=1, phi_dim=None, round_to_decimals=10, ignore_label_order=False, node_weight_function=None, use_early_stopping=True):
        self.h = h
        self.phi_dim = phi_dim
        self.round_to_decimals = round_to_decimals
        self.ignore_label_order = ignore_label_order
        self.node_weight_function = node_weight_function
        self.use_early_stopping = use_early_stopping

    def fit(self, X, y=None, **fit_params):
        assert len(X)
        X, node_weight_factors = _retrieve_node_weights_and_convert_graphs(X, node_weight_function=self.node_weight_function)

        # Remove empty graphs
        X = [x for x in X if x is not None]
        self.hashed_x = hash_dataset(X)

        phi_list, label_lookups, label_counters = fast_wl.transform(
            X,
            h=self.h,
            phi_dim=self.phi_dim,
            round_signatures_to_decimals=self.round_to_decimals,
            ignore_label_order=self.ignore_label_order,
            node_weight_factors=node_weight_factors,
            use_early_stopping=self.use_early_stopping
        )

        self.phi_list = [x.T for x in phi_list]
        self.phi_shape = self.phi_list[-1].shape
        self.label_lookups = label_lookups
        self.label_counters = label_counters

        return self

    def transform(self, X, y=None, **fit_params):
        assert len(X)
        X, node_weight_factors = _retrieve_node_weights_and_convert_graphs(X, node_weight_function=self.node_weight_function)

        # Use already computed phi_list if the given X is the same as in fit()
        if self.hashed_x == hash_dataset(X):
            return self.phi_list

        # Use early stopping
        h = min(len(self.phi_list) - 1, self.h)

        phi_list, label_lookups, label_counters = fast_wl.transform(
            X,
            h=h,
            phi_dim=self.phi_dim,
            round_signatures_to_decimals=self.round_to_decimals,
            ignore_label_order=self.ignore_label_order,
            node_weight_factors=node_weight_factors,
            use_early_stopping=False,
            append_to_labels=True,
            # Also give the label lookups/counters from training
            label_lookups=[dict(l) for l in self.label_lookups],
            label_counters=self.label_counters,
        )

        phi_list = [x.T for x in phi_list]

        # Do NOT save label lookups and counters! This would effectively be fitting!
        return phi_list


def hash_dataset(X):
    return ''.join([str(hash(''.join([str(a) for a in labels]))) for adj, labels in X])


def get_node_weight_factors(X, metric=None):
    if metric is None:
        return None

    out = [metric(x) for x in X]
    if isinstance(out[0], dict):
        out = [[int(val) for key, val in sorted(val.items(), key=lambda x: x[0])] for val in out]

    return out


def _retrieve_node_weights_and_convert_graphs(X, node_weight_function=None):
    X = graph_helper.get_graphs_only(X)
    node_weight_factors = get_node_weight_factors(X, metric=node_weight_function)
    X = graph_helper.convert_graphs_to_adjs_tuples(X, copy=True)
    return X, node_weight_factors
