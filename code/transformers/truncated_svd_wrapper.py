import sklearn
import sklearn.decomposition

class TruncatedSVDWrapper(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, percentage_n_components=0.5, n_iter=5):
        self.n_iter = n_iter
        self.percentage_n_components = percentage_n_components

    def fit(self, X, y=None, **fit_params):
        num_features = X.shape[1]
        n_components = int(self.percentage_n_components * num_features)
        self.trans_ = sklearn.decomposition.TruncatedSVD(n_components=n_components, n_iter=self.n_iter)
        self.trans_.fit(X.tocsr(), y)
        return self

    def transform(self, X, y=None, **fit_params):
        return self.trans_.transform(X.tocsr())
