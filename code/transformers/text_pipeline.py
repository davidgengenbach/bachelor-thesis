import sklearn
from sklearn import feature_extraction

def get_pipeline():
    return sklearn.pipeline.Pipeline([
        ('preprocessing', None),
        ('vectorizer', None),
        ('scaler', None),
        ('classifier', None)
    ])



def get_param_grid():
    return dict(
        vectorizer = [
            sklearn.feature_extraction.text.CountVectorizer(),
            sklearn.feature_extraction.text.TfidfVectorizer()
        ],
        vectorizer__stop_words = [None, 'english'],
        vectorizer__ngram_range = [(1, 1), (1, 2), (2, 2)]
        # preprocessing= [None, PreProcessingTransformer(only_nouns=True, return_lemma = True)],
        #scaler=[None, sklearn.preprocessing.StandardScaler(with_mean=False)]
    )