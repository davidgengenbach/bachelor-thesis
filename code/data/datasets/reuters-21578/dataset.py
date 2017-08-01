from nltk.corpus import reuters 

def fetch():
    categories = reuters.categories()
    X, Y = [], []
    for category in categories:
        cat_files = reuters.fileids(category)
        X += [reuters.raw(x) for x in cat_files]
        Y += [category] * len(cat_files)
    return X, Y