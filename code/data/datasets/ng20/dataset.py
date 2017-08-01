from sklearn.datasets import fetch_20newsgroups

def fetch(subset = 'all', remove = ('headers', 'footers', 'quotes'), categories = None):
    data = fetch_20newsgroups(subset = subset, remove = remove, categories = categories)
    target_names = data.target_names
    X = data.data
    Y = [target_names[target] for target in data.target]
    return X, Y
