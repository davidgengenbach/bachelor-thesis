import helper
import sklearn
import sklearn.metrics
import sys
import w2v_d2v


def get_classifiers(iterations=500):
    return {
        'PassiveAggressiveClassifier': sklearn.linear_model.PassiveAggressiveClassifier(),
        'Perceptron': sklearn.linear_model.Perceptron(n_iter=iterations),
        'LogisticRegression': sklearn.linear_model.LogisticRegression(max_iter=iterations),
        'SGDClassifier': sklearn.linear_model.SGDClassifier(n_iter=iterations)
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
