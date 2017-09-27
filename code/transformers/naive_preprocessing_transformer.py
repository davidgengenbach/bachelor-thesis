import preprocessing
import spacy
import sklearn

class NaivePreprocessingTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        def process(doc):
            return " ".join(x.text.strip() for x in doc).lower()
        result = [process(d) for d in preprocessing.get_spacy_parse(X)]
        return result

