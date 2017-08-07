import sklearn
import gensim

class Doc2VecTransformer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, embedding_size = 100, iterations = 10, infer_steps = 10):
        self.iterations = iterations
        self.embedding_size = embedding_size
        self.infer_steps = infer_steps

    def fit(self, X, Y, **fit_params):
        print('Doc2VecTransformer.fit: len(X)=len(Y)={}'.format(len(X)))
        tagged_documents = [gensim.models.doc2vec.TaggedDocument(
        words=text, tags=[tag]) for text, tag in zip(X, Y)]
        self.model = gensim.models.Doc2Vec(tagged_documents, size=self.embedding_size, iter=self.iterations)
        return self

    def transform(self, X, y=None, **fit_params):
        print('Doc2VecTransformer.transform: len(X)={}'.format(len(X)))
        return [self.model.infer_vector(x, steps=self.infer_steps) for x in X]

