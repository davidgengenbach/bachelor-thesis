from preprocessing import preprocessing
import sklearn

class SimplePreProcessingTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return [preprocessing.preprocess(d) for d in X]

