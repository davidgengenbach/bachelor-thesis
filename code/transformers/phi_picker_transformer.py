import sklearn
import scipy
from scipy import sparse

class PhiPickerTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, return_iteration=0, transpose=False, use_zeroth = True):

        assert return_iteration == 'stacked' or return_iteration >= -1

        self.return_iteration = return_iteration
        self.transpose = transpose
        self.use_zeroth = use_zeroth

    def fit(self, X, y=None, **fit_params):
        assert len(X)
        return self

    def transform(self, X, y=None, **fit_params):
        self.ensure_phi_iteration(X)

        if self.return_iteration == 'stacked':
            target = scipy.sparse.hstack(X if self.use_zeroth else X[1:])
        else:
            target = X[self.return_iteration]

        self.shape = target.shape

        return target.T if self.transpose else target


    def ensure_phi_iteration(self, X = None):
        if self.return_iteration != 'stacked' and isinstance(self.return_iteration, str):
            self.return_iteration = int(self.return_iteration)
        # X should be a list of phi matrices
        if X is not None:
            assert (self.return_iteration == 'stacked') or (self.return_iteration < len(X) or self.return_iteration == -1)