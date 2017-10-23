import sklearn
import scipy
from utils import graph_helper

class GraphToTextTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, use_edges = True):
        self.use_edges = use_edges

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        graph_helper.remove_graph_labels(X)
        X = [graph_helper.graph_to_text(g, self.use_edges) for g in X]
        return X