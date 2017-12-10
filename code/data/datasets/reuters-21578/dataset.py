import nltk
from utils import dataset_helper

def fetch():
    nltk.download('reuters')
    from nltk.corpus import reuters

    X, Y = [], []
    for file_id in reuters.fileids():
        classes = reuters.categories([file_id])
        if len(classes) != 1: continue
        X.append(reuters.raw(file_id))
        Y.append(classes[0])

    X, Y = dataset_helper.filter_out_text_with_less_words(X, Y, min_words=20)
    return X, Y