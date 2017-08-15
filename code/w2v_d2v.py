import gensim
import sklearn
from sklearn import metrics
import numpy as np
from nltk.stem import PorterStemmer, WordNetLemmatizer

porter = PorterStemmer()
wnl = WordNetLemmatizer()


def lemmatize(ambiguous_word, pos=None, neverstem=True,
              lemmatizer=wnl, stemmer=porter):
    """Tries to convert a surface word into lemma, and if lemmatize word is not in
    wordnet then try and convert surface word into its stem.
    This is to handle the case where users input a surface word as an ambiguous 
    word and the surface word is a not a lemma.

    Adapted from https://github.com/alvations/pywsd/blob/master/pywsd/utils.py
    """
    if pos:
        lemma = lemmatizer.lemmatize(ambiguous_word, pos=pos)
    else:
        lemma = lemmatizer.lemmatize(ambiguous_word)
    stem = stemmer.stem(ambiguous_word)
    # Ensure that ambiguous word is a lemma.
    if not wn.synsets(lemma):
        if neverstem:
            return ambiguous_word
        if not wn.synsets(stem):
            return ambiguous_word
        else:
            return stem
    else:
        return lemma

tokenizer = sklearn.feature_extraction.text.CountVectorizer(stop_words='english').build_tokenizer()

def w2v_preproess(doc, lemmatize_word=False):
    return [lemmatize(x.lower()) if lemmatize_word else x.lower() for x in tokenizer(doc)]

def init_w2v_google(googlenews_vector_file = 'data/GoogleNews-vectors-negative300.bin'):
    model = gensim.models.KeyedVectors.load_word2vec_format(googlenews_vector_file, binary=True)
    return model

def train_w2v(w2v_data, iterations=50):
    model_w2v = gensim.models.Word2Vec(w2v_data, iter=iterations)
    return model_w2v


def train_d2v(w2v_data, labels, iterations=50):
    tagged_documents = [gensim.models.doc2vec.TaggedDocument(
        words=words, tags=[tag]) for words, tag in zip(w2v_data, labels)]
    model_d2v = gensim.models.Doc2Vec(tagged_documents, size=1000, iter=iterations)
    return model_d2v


def fit_and_predict_d2v(clf, model_d2v, d2v_inferred_train, d2v_inferred_test):
    """Fit the given classifier to the train/test data.
        Return predictions.
        """
    clf.fit(model_d2v.docvecs, [model_d2v.docvecs.index_to_doctag(idx) for idx in range(len(model_d2v.docvecs))])
    pred_train = clf.predict(np.array(d2v_inferred_train))
    pred_test = clf.predict(np.array(d2v_inferred_test))
    return pred_train, pred_test


def score_d2v(clfs, targets_train, target_test, model_d2v, w2v_data, w2v_data_test, steps=10):
    d2v_inferred_train = [model_d2v.infer_vector(x, steps=steps) for x in w2v_data]
    print('Inferred train vectors')
    d2v_inferred_test = [model_d2v.infer_vector(x, steps=steps) for x in w2v_data_test]
    print('Inferred test vectors')
    d2v_classification_predictions = {clf_name: fit_and_predict_d2v(
        clf, model_d2v, d2v_inferred_train, d2v_inferred_test) for clf_name, clf in clfs.items()}
    results = {}

    for clf_name, predictions in d2v_classification_predictions.items():
        pred_train, pred_test = predictions[0], predictions[1]
        f1_score_train = metrics.f1_score(targets_train, pred_train, average='macro')
        f1_score_test = metrics.f1_score(target_test, pred_test, average='macro')
        confusion_matrix = metrics.confusion_matrix(target_test, pred_test)
        results[clf_name] = {'train': f1_score_train, 'test': f1_score_test}
    return results
