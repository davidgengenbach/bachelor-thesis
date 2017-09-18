from glob import glob
import os
import importlib
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle
from joblib import Parallel, delayed
from collections import Counter
import numpy as np
import pandas as pd
import graph_helper
import scipy.sparse

DATASETS_LIMITED = ['ng20', 'reuters-21578', 'webkb', 'ling-spam']

PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = 'data/datasets'
GRAPHS_FOLDER = 'data/graphs'
CACHE_PATH = 'data/CACHE'

def split_dataset(X, Y, train_size=0.8, random_state_for_shuffle=42):
    """Returns a train/test split for the given dataset.

    Args:
        X (list): The docs/graphs
        Y (list of str): The labels
        train_size (float, optional): from 0.0 to 1.0, which percentage is used for train
        random_state_for_shuffle (int, optional): Description

    Returns:
        tuple(list): four array corresponding to:
            data_train_X, data_test_X, data_train_Y, data_test_Y
    """
    if train_size != 1.0:
        return train_test_split(
            X, Y,
            train_size=train_size,
            random_state=random_state_for_shuffle,
            stratify=Y
        )
    else:
        X_, Y_ = shuffle(
            X, Y,
            random_state=random_state_for_shuffle
        )
        return X_, [], Y_, []

def preprocess(X, n_jobs=4, **kwargs):
    import preprocessing
    return Parallel(n_jobs=n_jobs)(delayed(preprocessing.preprocess_text)(text, **kwargs) for text in X)


def get_dataset_module(dataset_folder, dataset_name):
    """Returns the dataset python module for a given dataset.
    Searches for the module definition in:
    - {dataset_folder}/{dataset_name}/dataset.py
    - {dataset_folder}/dataset_{dataset_name}.py

    Args:
        dataset_folder (str): where to search for the dataset py
        dataset_name (str): which dataset to search for

    Returns:
        module: the dataset module that has a fetch method
    """

    def import_from_path(path): return importlib.import_module(path.replace('/', '.'))

    dataset_module_onefile = os.path.join(dataset_folder, 'dataset_' + dataset_name)
    dataset_module_in_folder = os.path.join(dataset_folder, dataset_name, 'dataset')

    candidates = [dataset_module_onefile, dataset_module_in_folder]

    dataset_module = None

    for candidate in candidates:
        if os.path.exists(candidate + '.py'):
            dataset_module = import_from_path(candidate)
            break
    assert dataset_module, 'Could not find dataset definition for dataset "{}" in: {}'.format(dataset_name, candidates)
    assert hasattr(dataset_module, 'fetch'), 'Dataset {} does not have a fetch method'.format(dataset_name)
    return dataset_module


def test_dataset_validity(X, Y):
    def sparse_matrix_type(a):
        types = [scipy.sparse.lil_matrix, scipy.sparse.coo_matrix]
        for t in types:
            if isinstance(a, t): return True
        return False
    assert isinstance(X, list) or sparse_matrix_type(X), 'X must be a list, it is: {}'.format(type(X))
    assert isinstance(Y, list), 'Y must be a list'
    if sparse_matrix_type(X):
        assert X.shape[0] and len(Y), 'Dataset is empty'
        assert X.shape[0] == len(Y), 'X and Y must have same length'
    else:
        assert len(X) and len(Y), 'Dataset is empty'
        assert len(X) == len(Y), 'X and Y must have same length'
    assert len(set(Y)), 'Y must contain at least one label'


def get_dataset_cached(cache_file, check_validity = True):
    assert os.path.exists(cache_file), 'Could not find cache_file: {}'.format(cache_file)
    with open(cache_file, 'rb') as f:
        X, Y = pickle.load(f)
    if check_validity:
        test_dataset_validity(X, Y)
    return X, Y

def get_dataset(dataset_name, use_cached=True, preprocessed=False, dataset_folder=DATASET_FOLDER, preprocessing_args=None, cache_path=CACHE_PATH, transform_fn=None, cache_file=None):
    """Returns the dataset as a tuple of lists. The first list contains the data, the second the labels
    TODO: Caching could be done with decorator.

    Args:
        dataset_name (str): the name of the dataset
        use_cached (bool, optional): whether to use the cached dataset
        preprocessed (bool, optional): whether to preprocess the text
        dataset_folder (str, optional): where to search the dataset
        preprocessing_args (None, optional)
        cache_path (str, optional): folder to save the dataset to after fetching it
        transform_fn (function, optional): gets applied to the dataset before saving it (only when use_cache=False or the dataset_npy does not exist!)
        cache_file (str, optional): used to overwrite the cache-path

    Returns:
        tuple(list): two lists, the first the data, the second the labels
    """

    if cache_file:
        dataset_npy = cache_file
    else:
        dataset_npy = os.path.join(cache_path, 'dataset_{}.npy'.format(dataset_name))

    if use_cached and os.path.exists(dataset_npy):
        X, Y = get_dataset_cached(dataset_npy)
    else:
        X, Y = get_dataset_module(dataset_folder, dataset_name).fetch()

        # Test dataset validity before saving it
        test_dataset_validity(X, Y)

        if transform_fn:
            X, Y = transform_fn(X, Y)

        with open(dataset_npy, 'wb') as f:
            pickle.dump((X, Y), f)

    test_dataset_validity(X, Y)
    return X, Y


def get_gml_graph_dataset(dataset_name, use_cached=True, graphs_folder=GRAPHS_FOLDER, cache_folder=CACHE_PATH):
    """Retrieves the gml dataset.
    TODO: The caching could be done with a decorator.

    Args:
        dataset_name (str): the dataset
        use_cached (bool, optional): 
        graphs_folder (str, optional): where to search the gml graphs
        cache_folder (str, optional): where to cache

    Returns:
        tuple(list, list): X and Y
    """
    graph_folder = os.path.join(graphs_folder, dataset_name)
    cache_npy = os.path.join(CACHE_PATH, 'dataset_graph_gml_{}.npy'.format(dataset_name))

    if use_cached and os.path.exists(cache_npy):
        with open(cache_npy, 'rb') as f:
            X, Y = pickle.load(f)
    else:
        X, Y = graph_helper.get_graphs_from_folder(graph_folder, undirected=True)
        # Test dataset validity before saving it
        test_dataset_validity(X, Y)

        with open(cache_npy, 'wb') as f:
            pickle.dump((X, Y), f)
    return X, Y

def get_subset_with_most_frequent_classes(X, Y, num_classes_to_keep = 2, use_numpy = False):
    most_common_classes = [label for label, count in Counter(Y).most_common(num_classes_to_keep)]
    if use_numpy:
        # A little slower
        indices = np.in1d(Y, most_common_classes)
        return np.array(X, dtype=str)[indices], np.array(Y, dtype=str)[indices]
    else:
        most_common_classes = set(most_common_classes)
        data = [(text, label) for text, label in zip(X, Y) if label in most_common_classes]
        return [x[0] for x in data], [x[1] for x in data]


def get_dataset_subset_with_most_frequent_classes(dataset_name, num_classes_to_keep=2, get_dataset_kwargs = {}):
    """Given a base dataset, this function returns the num_classes_to_keep classes.

    Args:
        dataset_name (str): base dataset
        num_classes_to_keep (int, optional): how many classes should be taken
        use_numpy (bool, optional): slows down a little

    Returns:
        tuple(list, list): X and Y, where the dataset only contains data from the most frequent classes
    """
    X, Y = get_dataset(dataset_name, **get_dataset_kwargs)
    return get_subset_with_most_frequent_classes(X, Y)


def get_all_available_dataset_names(dataset_folder=DATASET_FOLDER):
    """Searches for available datasets.

    Args:
        dataset_folder (str, optional): where to search

    Returns:
        list(str): a list of strings that can be retrieved through `get_dataset`
    """
    datasets = glob('{}/*/dataset.py'.format(dataset_folder))
    datasets += glob('{}/dataset_*.py'.format(dataset_folder))

    dataset_folders = [
        x.replace('/dataset.py', '').replace('dataset_', '').replace('.py', '').replace('/', '.').split('.')[-1]
        for x in datasets
    ]
    return sorted(dataset_folders)

def get_all_cached_datasets(cache_path = CACHE_PATH):
    return sorted(glob(cache_path + '/*.npy'))

def get_all_cached_graph_datasets(dataset_name = None, cache_path = CACHE_PATH):
    return [x for x in get_all_cached_datasets(cache_path) if x.split('/')[-1].startswith('dataset_graph') and '_relabeled' not in x and 'phi' not in x and (not dataset_name or get_dataset_name_from_graph_cachefile(x) == dataset_name or get_dataset_name_from_graph_cachefile(x) == dataset_name + '-single')]

def get_all_cached_graph_phi_datasets(dataset_name = None, cache_path = CACHE_PATH):
    if dataset_name:
        dataset_name = dataset_name.replace('-single', '')
    return [x for x in get_all_cached_datasets(cache_path) if 'phi' in x and (not dataset_name or get_dataset_name_from_graph_cachefile(x) == dataset_name)]

def get_dataset_name_from_graph_cachefile(graph_cache_file, replace_single = True):
    dataset = graph_cache_file.rsplit('.npy')[0].split('/')[-1]
    dataset = dataset.rsplit('_', 1)[1].replace('.phi', '')
    if replace_single:
        dataset = dataset.replace('-single', '')
    return dataset

def get_all_datasets(dataset_folder=DATASET_FOLDER, **kwargs):
    """Returns a dict with the available datasets as key and the documents as values

    Args:
        dataset_folder (str, optional): Where to search for datasets

    Returns:
        dict: Keys are the dataset names, the values is a list of docs like [(topic1, document1], (topic2, document2))]
    """
    return {dataset: get_dataset(dataset, **kwargs) for dataset in get_all_available_dataset_names(dataset_folder)}


def get_w2v_embedding_for_dataset(dataset_name, embedding_folder = 'data/embeddings/trained'):
    embedding_file = os.path.join(embedding_folder, dataset_name + '.npy')
    assert os.path.exists(embedding_file)
    with open(embedding_file, 'rb') as f:
        return pickle.load(f)


def get_dataset_dict(X, Y=None):
    """Returns a dictionary where the keys are the topics, the values are the documents of that topic. 

    Args:
        X (list of list of str): The documents
        Y (list of str): The topics for the topics. len(X) == len(Y). If Y is None, X must be a list of tuples

    Returns:
        dict: Keys are topics, values are the corresponding docs
    """
    if not Y:
        data = X
        Y = [x[0] for x in data]
        X = [x[1] for x in data]

    assert len(X) == len(Y), 'X and Y do not have the same length.'
    assert len(set(Y)) > 0, 'No classes.'

    topics = {x: [] for x in set(Y)}
    for clazz, doc in zip(Y, X):
        topics[clazz].append(doc)
    return topics


def plot_dataset_class_distribution(X, Y, title='Docs per topic', figsize=(14, 8), ax = None, log = True):
    x_per_topic = get_dataset_dict(X, Y)
    df_graphs_per_topic = pd.DataFrame([
        (topic, len(docs)) for topic, docs in x_per_topic.items()],
        columns=['topic', 'num_docs']
    ).set_index(['topic']).sort_values(by='num_docs')
    ax = df_graphs_per_topic.plot.barh(title='Docs per topic', legend=False, figsize=figsize, ax = ax, log = log)
    ax.set_xlabel('# docs')
    return ax
