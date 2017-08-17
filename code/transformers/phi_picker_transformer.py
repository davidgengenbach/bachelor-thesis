import sklearn


class PhiPickerTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, return_iteration=0, transpose=True):
        assert return_iteration >= -1
        self.return_iteration = return_iteration
        self.transpose = False

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        # X should be a list of phi matrices
        assert self.return_iteration < len(X) or self.return_iteration == -1
        target = X[self.return_iteration]
        return target.T if transpose else target
