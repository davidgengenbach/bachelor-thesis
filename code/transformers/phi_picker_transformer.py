import sklearn
import scipy
from scipy import sparse

class PhiPickerTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, return_iteration=0, transpose=False):
        if return_iteration != 'stacked' and isinstance(return_iteration, str):
            return_iteration = int(return_iteration)

        assert return_iteration == 'stacked' or return_iteration >= -1

        self.return_iteration = return_iteration
        self.transpose = transpose

    def fit(self, X, y=None, **fit_params):
        assert len(X)
        return self

    def transform(self, X, y=None, **fit_params):
        # X should be a list of phi matrices
        assert (self.return_iteration == 'stacked') or (self.return_iteration < len(X) or self.return_iteration == -1)

        if self.return_iteration == 'stacked':
            target = scipy.sparse.hstack(X)
        else:
            target = X[self.return_iteration]
        return target.T if self.transpose else target
