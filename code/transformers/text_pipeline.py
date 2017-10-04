import sklearn
from sklearn import feature_extraction

def get_pipeline():
    return sklearn.pipeline.Pipeline([
        ('preprocessing', None),
        ('TfidfTransformer', sklearn.feature_extraction.text.TfidfVectorizer()),
        ('scaler', None),
        ('clf', None)
    ])

def get_param_grid():
    return dict(
        TfidfTransformer__stop_words = [None, 'english'],
        # preprocessing= [None, PreProcessingTransformer(only_nouns=True, return_lemma = True)],
        #scaler=[None, sklearn.preprocessing.StandardScaler(with_mean=False)]
    )