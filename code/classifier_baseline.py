import helper
import sklearn
import sklearn.metrics

def vectorize_text(data_train_X, data_test_X, stopwords = 'english'):
    # CountVectorizer
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(stop_words = 'english')
    vectorizer.fit(data_train_X)

    vectors_train = vectorizer.transform(data_train_X)
    vectors_test = vectorizer.transform(data_test_X)

    # TfidfTransformer
    tfidf_transformer = sklearn.feature_extraction.text.TfidfTransformer()
    tfidf_transformer.fit(vectors_train)
    vectors_trans_train = tfidf_transformer.transform(vectors_train)
    vectors_trans_test = tfidf_transformer.transform(vectors_test)
    return vectors_trans_train, vectors_trans_test

def get_predictions_for_classifiers(vectors_trans_train, vectors_trans_test, data_train_Y, data_test_Y, clfs = helper.get_classifiers(iterations = 1)):
    results = {}
    for clf_name, clf in clfs.items():
        print('Classifier: {}'.format(clf_name))
        # Fit
        clf.fit(vectors_trans_train, data_train_Y)
        
        # Predict
        pred_train = clf.predict(vectors_trans_train)
        pred_test = clf.predict(vectors_trans_test)
        
        # Metric
        f1_score_train = sklearn.metrics.f1_score(data_train_Y, pred_train, average='macro')
        f1_score_test = sklearn.metrics.f1_score(data_test_Y, pred_test, average='macro')
        results[clf_name] = {'train': f1_score_train, 'test': f1_score_test, 'acc': sklearn.metrics.accuracy_score(data_test_Y, pred_test)}
    return results