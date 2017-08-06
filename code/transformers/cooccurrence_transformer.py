import cooccurence

class CoOccurrenceTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Co-Occurrence transformer. Takes texts or spacy docs as input and transforms it into networkx graphs (co-occurence).
    """

    def __init__(self, window_size = 2, ignore_sentence_bounds = True, only_forward_window = False):
        self.window_size = window_size
        self.ignore_sentence_bounds = ignore_sentence_bounds
        self.only_forward_window = only_forward_window

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        def process(d):
            return cooccurence.get_coocurrence_matrix(d, self.window_size, ign)


        return [process(d) for d in X]

