import sklearn

class PhiListToGramMatrixTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        assert len(X)
        K = X[0].dot(X[0].T)
        for x in X[1:]:
            K = K + x.dot(x.T)
        return K