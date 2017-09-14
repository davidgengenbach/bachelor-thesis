import graph_helper
import sklearn

class NxGraphToTupleTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, from_nx_to_tuple = True):
        self.from_nx_to_tuple = from_nx_to_tuple
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        if self.from_nx_to_tuple:
            graph_helper.convert_graphs_to_adjs_tuples(X)
        else:
            graph_helper.convert_adjs_tuples_to_graphs(X)
        return X

