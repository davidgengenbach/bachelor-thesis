import graph_helper
import sklearn

class RelabelGraphsTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, lookup = None):
        self.lookup = lookup

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        for idx, (adj, nodes) in enumerate(X):
            relabeled_nodes = [self.lookup.get(label, label) for label in nodes]
            X[idx] = (adj, relabeled_nodes)
        return X

