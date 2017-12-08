import sklearn


class RelabelGraphsTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, lookup = None):
        self.lookup = lookup

    def fit(self, X, y=None, **fit_params):
        # TODO: find low-frequency node labels, merge them with other labels
        return self

    def transform(self, X, y=None, **fit_params):
        out = []
        for idx, (adj, nodes) in enumerate(X):
            relabeled_nodes = [str(self.lookup.get(label, label)).strip() for label in nodes]
            out.append((adj, relabeled_nodes))
        return out

