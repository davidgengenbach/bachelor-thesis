import helper
import sklearn
import sklearn.metrics
import sys
import w2v_d2v


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


class MostFrequentLabelClassifier(object):
    """Baseline classifier that always returns the label that was most frequent in the training set.

    Attributes:
        label_to_return (str): the label that will be predicted
    """

    def __init__(self):
        pass

    def fit(self, x=None, y=None):
        label_counts = {}
        for y_ in y:
            if y_ not in label_counts:
                label_counts[y_] = 0
            label_counts[y_] += 1
        self.label_to_return = max(label_counts.items(), key=lambda x: x[1])[0]

    def predict(self, x=None):
        return [self.label_to_return] * x.shape[0]


def get_classifiers(iterations=500):
    return {
        'PassiveAggressiveClassifier': sklearn.linear_model.PassiveAggressiveClassifier(),
        'Perceptron': sklearn.linear_model.Perceptron(n_iter=iterations),
        'LogisticRegression': sklearn.linear_model.LogisticRegression(max_iter=iterations),
        'SGDClassifier': sklearn.linear_model.SGDClassifier(n_iter=iterations),
        'MostFrequentLabel': MostFrequentLabelClassifier(),
        #'Doc2VecClassifier': Doc2VecClassifier(n_iter = iterations),
        #'kNN': sklearn.neighbors.KNeighborsClassifier()
    }

def vectorize_text(data_train_X, data_test_X, stopwords='english'):
    # CountVectorizer
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words='english')
    vectorizer.fit(data_train_X)

    vectors_train = vectorizer.transform(data_train_X)
    vectors_test = vectorizer.transform(data_test_X)

    # TfidfTransformer
    tfidf_transformer = sklearn.feature_extraction.text.TfidfTransformer()
    tfidf_transformer.fit(vectors_train)
    vectors_trans_train = tfidf_transformer.transform(vectors_train)
    vectors_trans_test = tfidf_transformer.transform(vectors_test)
    return vectors_trans_train, vectors_trans_test

def get_f1_score_for_clfs(pred_results, data_train_Y, data_test_Y):
    results = {}
    for clf_name, values in pred_results.items():
        pred_train_Y, pred_test_Y = values['train'], values['test']
        f1_score_train, f1_score_test = get_f1_score(pred_train_Y, pred_test_Y, data_train_Y, data_test_Y)
        results[clf_name] = {'train': f1_score_train, 'test': f1_score_test}
    return results

def get_f1_score(pred_train_Y, pred_test_Y, data_train_Y, data_test_Y):
    # Metric
    f1_score_train = sklearn.metrics.f1_score(data_train_Y, pred_train_Y, average='macro')
    f1_score_test = sklearn.metrics.f1_score(data_test_Y, pred_test_Y, average='macro')
    return f1_score_train, f1_score_test

def get_predictions_for_classifiers(vectors_trans_train, vectors_trans_test, data_train_Y, data_test_Y, data_train_X, data_test_X, clfs=get_classifiers(iterations=1)):
    results = {}
    for idx, (clf_name, clf) in enumerate(clfs.items()):
        sys.stdout.write('\rCurrently training classifier: {:<30} ({}/{})'.format(clf_name, idx + 1, len(clfs.keys())))
        train_X, test_X = vectors_trans_train, vectors_trans_test
        if hasattr(clf, 'needs_raw_doc') and clf.needs_raw_doc:
            train_X, test_X = data_train_X, data_test_X
        # Fit
        clf.fit(train_X, data_train_Y)

        # Predict
        pred_train_Y = clf.predict(train_X)
        pred_test_Y = clf.predict(test_X)

        results[clf_name] = {'train': pred_train_Y, 'test': pred_test_Y}
    print()
    return results
