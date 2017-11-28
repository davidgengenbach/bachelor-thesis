import sklearn

from utils import graph_helper


class NxGraphToTupleTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, from_nx_to_tuple = True):
        self.from_nx_to_tuple = from_nx_to_tuple

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if self.from_nx_to_tuple:
            X = graph_helper.get_graphs_only(X)
            graph_helper.convert_graphs_to_adjs_tuples(X)
        else:
            graph_helper.convert_adjs_tuples_to_graphs(X)
        return X
