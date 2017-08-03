from glob import glob
import os
import importlib
import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle
from joblib import Parallel, delayed
from collections import Counter
import numpy as np
import pandas as pd

PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = 'data/datasets'


def split_dataset(X, Y, train_size=0.8, random_state_for_shuffle=42):
    if train_size != 1.0:
        return train_test_split(
            X, Y,
            train_size=train_size,
            random_state=random_state_for_shuffle,
            stratify=Y
        )
    else:
        return shuffle(
            X, Y,
            random_state=random_state_for_shuffle
        ), []


def preprocess(X, n_jobs=4, **kwargs):
    return Parallel(n_jobs=n_jobs)(delayed(preprocessing.preprocess_text)(text, **kwargs) for text in X)


def get_dataset_dict_preprocessed(dataset_name, dataset_folder=DATASET_FOLDER, use_cached=True):
    X, Y = get_dataset(dataset_name, dataset_folder=dataset_folder, use_cached=use_cached)
    X = preprocess(X)
    return X, Y


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


def get_dataset(dataset_name, use_cached=True, preprocessed=False, dataset_folder=DATASET_FOLDER, preprocessing_args=None, cache_path='data/CACHE'):
    """Returns the dataset as a list of docs with labels: [(topic1, document1], (topic2, document2))]
    
    Args:
        dataset_name (str): The name of the dataset
        use_cached (bool, optional): Whether to use the cached dataset
        preprocessed (bool, optional): 
        dataset_folder (str, optional): Where to search the dataset
        preprocessing_args (None, optional): 
        cache_path (str, optional): folder to save the dataset to after fetching it
    
    Returns:
        list(tuples): a list of documents with labels
    """
    def test_dataset(X, Y):
        assert isinstance(X, list), 'X must be a list'
        assert isinstance(Y, list), 'Y must be a list'
        assert len(X) and len(Y), 'Dataset is empty'
        assert len(X) == len(Y), 'X and Y must have same length'

    dataset_npy = os.path.join(cache_path, 'dataset_{}.npy'.format(dataset_name))

    if use_cached and os.path.exists(dataset_npy):
        with open(dataset_npy, 'rb') as f:
            X, Y = pickle.load(f)
    else:
        X, Y = get_dataset_module(dataset_folder, dataset_name).fetch()

        # Test dataset validity before saving it
        test_dataset(X, Y)

        with open(dataset_npy, 'wb') as f:
            pickle.dump((X, Y), f)

    test_dataset(X, Y)

    if preprocessed:
        X = preprocess(X, **preprocessing_args)
    return X, Y


def get_dataset_subset_with_most_frequent_classes(dataset_name, num_classes_to_keep=2):
    X, Y = get_dataset(dataset_name)
    most_common_classes = [label for label, count in Counter(Y).most_common(num_classes_to_keep)]
    if False:
        indices = np.in1d(Y, most_common_classes)
        return np.array(X, dtype=str)[indices], np.array(Y, dtype=str)[indices]
    else:
        data = [(text, label) for text, label in zip(X, Y) if label in most_common_classes]
        return [x[0] for x in data], [x[1] for x in data]


def get_all_datasets(dataset_folder=DATASET_FOLDER, **kwargs):
    """Returns a dict with the available datasets as key and the documents as values

    Args:
        dataset_folder (str, optional): Where to search for datasets

    Returns:
        dict: Keys are the dataset names, the values is a list of docs like [(topic1, document1], (topic2, document2))]
    """
    datasets = glob('{}/*/dataset.py'.format(dataset_folder)) + glob('{}/dataset_*.py'.format(dataset_folder))
    dataset_folders = [x.replace('/dataset.py', '').replace('dataset_', '').replace('.py', '').replace('/', '.') for x in datasets]
    return {x.split('.')[-1]: get_dataset(x.split('.')[-1], **kwargs) for x in dataset_folders}


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

    assert len(X) == len(Y)
    assert len(set(Y)) > 0

    topics = {x: [] for x in set(Y)}
    for clazz, words in zip(Y, X):
        if words.strip() != '':
            topics[clazz].append(words)
    return topics


def plot_dataset_class_distribution(X, Y, figsize=(14, 8)):
    x_per_topic = get_dataset_dict(X, Y)
    df_graphs_per_topic = pd.DataFrame([(topic, len(docs)) for topic, docs in x_per_topic.items()], columns=[
                                       'topic', 'num_docs']).set_index(['topic']).sort_values(by='num_docs')
    ax = df_graphs_per_topic.plot.barh(title='Docs per topic', legend=False, figsize=figsize)
    ax.set_xlabel('# docs')
    return ax

if __name__ == '__main__':
    get_all_datasets()
