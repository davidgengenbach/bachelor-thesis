import sklearn

class PhiPickerTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, return_iteration = 0):
        assert return_iteration >= -1
        self.return_iteration = return_iteration

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        # X should be a list of phi matrices
        assert self.return_iteration < len(X) or self.return_iteration == -1
        return X[self.return_iteration].T
