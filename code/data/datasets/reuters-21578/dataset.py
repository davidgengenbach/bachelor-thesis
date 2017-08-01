from nltk.corpus import reuters 

def fetch():
    return get_dataset()

def get_dataset():
    categories = reuters.categories()
    X, Y = [], []
    for cat in categories:
        cat_files = reuters.fileids(cat)
        X += [reuters.raw(x) for x in cat_files]
        Y += cat
    return X, Y