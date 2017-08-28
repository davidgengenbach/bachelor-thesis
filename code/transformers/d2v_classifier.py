class Doc2VecClassifier(object):
    needs_raw_doc = True

    def __init__(self, n_iter = 1, prediction_steps = 1, preprocess_data = True, infer_steps = 1):
        self.n_iter = n_iter
        self.prediction_steps = prediction_steps
        self.preprocess_data = preprocess_data
        self.infer_steps = infer_steps

    def preprocess(self, X):
        if self.preprocess_data:
            return [w2v_d2v.w2v_preproess(doc) for doc in X]
        else:
            return X

    def fit(self, X, Y):
        self.model = w2v_d2v.train_d2v(self.preprocess(X), Y, iterations = self.n_iter)

    def predict(self, X):
        inferred = [self.model.infer_vector(x, steps=self.infer_steps) for x in self.preprocess(X)]
        return self.model.predict(np.array(inferred))
