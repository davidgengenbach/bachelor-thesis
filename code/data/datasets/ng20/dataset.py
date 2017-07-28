def fetch():
    return get_topics_from_sklearn()

def get_topics_from_sklearn(subset = 'all', remove = ('headers', 'footers', 'quotes'), categories = None):
    from sklearn.datasets import fetch_20newsgroups
    data = fetch_20newsgroups(subset = subset, remove = remove, categories = categories)
    target_names = data.target_names
    return [(target_names[target], doc) for target, doc in zip(data.target, data.data)]