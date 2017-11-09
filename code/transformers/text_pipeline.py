import sklearn
from sklearn import feature_extraction
from sklearn import preprocessing
from transformers.preprocessing_transformer import PreProcessingTransformer
from transformers.simple_preprocessing_transformer import SimplePreProcessingTransformer

def get_pipeline():
    return sklearn.pipeline.Pipeline([
        ('preprocessing', None),
        ('vectorizer', None),
        ('scaler', None),
        ('classifier', None)
    ])


def get_param_grid(reduced = False):
    params = dict(
        vectorizer = [
            sklearn.feature_extraction.text.CountVectorizer(),
            sklearn.feature_extraction.text.TfidfVectorizer()
        ],
        vectorizer__stop_words = [None, 'english'],
        vectorizer__ngram_range = [(1, 1), (1, 2), (2, 2)],
        vectorizer__binary = [True, False],
        #preprocessing= [None, PreProcessingTransformer(only_nouns=True, return_lemma = True)],
        preprocessing = [None, SimplePreProcessingTransformer()],
        #scaler=[None, sklearn.preprocessing.MaxAbsScaler()]
    )

    if reduced:
        params['vectorizer'] = [sklearn.feature_extraction.text.CountVectorizer()]
        params['vectorizer__ngram_range'] = [(1, 2)]
        params['vectorizer__stop_words'] = ['english']
        params['preprocessing'] = [None]

    return params