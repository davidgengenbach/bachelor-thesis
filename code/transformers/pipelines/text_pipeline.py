import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers.preprocessing_transformer import PreProcessingTransformer
from transformers.simple_preprocessing_transformer import SimplePreProcessingTransformer

_preprocessors = [
    None,
    PreProcessingTransformer(only_nouns=True, return_lemma=True),
    SimplePreProcessingTransformer()
]

def get_params(reduced = False):
    pipeline = sklearn.pipeline.Pipeline([
        ('preprocessing', None),
        ('vectorizer', None),
        ('scaler', None),
        ('classifier', None)
    ])

    params = dict(
        vectorizer=[
            CountVectorizer(),
            TfidfVectorizer()
        ],
        vectorizer__ngram_range=[(1, 1), (1, 2), (2, 2)],
        vectorizer__binary=[True, False],
        preprocessing=[None, SimplePreProcessingTransformer()],
    )

    if reduced:
        params['vectorizer'] = [CountVectorizer()]
        params['vectorizer__ngram_range'] = [(1, 1)]
        params['vectorizer__binary'] = [True]
        params['preprocessing'] = [None]

    return pipeline, params
