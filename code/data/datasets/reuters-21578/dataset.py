from nltk.corpus import reuters 

def fetch():
    return get_dataset()

def get_dataset():
    cats = reuters.categories()
    data = []
    for cat in cats:
        cat_files = reuters.fileids(cat)
        data += [(cat, reuters.raw(x)) for x in cat_files]
    return data