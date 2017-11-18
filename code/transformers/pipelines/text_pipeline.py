import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers.simple_preprocessing_transformer import SimplePreProcessingTransformer

def get_params(reduced = False):
    pipeline = sklearn.pipeline.Pipeline([
        ('preprocessing', None),
        ('vectorizer', None),
        ('scaler', None)
    ])

    params = dict(
        vectorizer=[
            CountVectorizer(),
            TfidfVectorizer()
        ],
        vectorizer__ngram_range=[(1, 1), (1, 2)],
        vectorizer__binary=[True, False],
        preprocessing=[SimplePreProcessingTransformer()],
    )

    if reduced:
        params['vectorizer'] = [CountVectorizer()]
        params['vectorizer__ngram_range'] = [(1, 1)]
        params['vectorizer__binary'] = [True]
        params['preprocessing'] = [None]

    return pipeline, params
