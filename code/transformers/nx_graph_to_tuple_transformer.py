import graph_helper
import sklearn

class NxGraphToTupleTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        graph_helper.convert_graphs_to_adjs_tuples(X)
        return X

