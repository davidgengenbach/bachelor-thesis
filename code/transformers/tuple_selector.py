import sklearn
from sklearn import base
import scipy
from scipy import sparse

class TupleSelector(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, tuple_index=0, v_stack=False):
        self.tuple_index = tuple_index
        self.v_stack = v_stack

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        data = [x[self.tuple_index] for x in X]
        return scipy.sparse.vstack(data) if self.v_stack else data
