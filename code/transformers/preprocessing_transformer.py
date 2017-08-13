import preprocessing
import spacy
import sklearn

class PreProcessingTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, only_nouns = False, return_text = True):
        self.only_nouns = only_nouns
        self.return_text = return_text

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        def process(doc):
            processed_words = [word for word in doc if (not self.only_nouns or word.pos == spacy.symbols.NOUN)]
            return " ".join(word.text for word in processed_words) if self.return_text else processed_words
        return [process(d) for d in preprocessing.get_spacy_parse(X)]
