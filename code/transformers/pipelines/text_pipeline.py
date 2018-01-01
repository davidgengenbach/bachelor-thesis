import sklearn
import sklearn.preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers.simple_preprocessing_transformer import SimplePreProcessingTransformer

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
        vectorizer__ngram_range=[(1, 1), (1, 2)],
        vectorizer__binary=[True, False],
        vectorizer__min_df=[2],
        preprocessing=[SimplePreProcessingTransformer()],
        scaler=[sklearn.preprocessing.MaxAbsScaler()]
    )

    if reduced:
        params['vectorizer'] = [CountVectorizer()]
        params['vectorizer__ngram_range'] = [(1, 1)]
        params['vectorizer__binary'] = [True, False]

    return pipeline, params
