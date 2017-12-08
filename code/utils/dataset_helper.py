from utils.constants import *
import importlib
import os
import pickle
from collections import Counter
from glob import glob

import numpy as np
import scipy.sparse
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from utils import filename_utils, graph_helper


def get_dataset(dataset_name, use_cached=True, dataset_folder=DATASET_FOLDER, cache_path=CACHE_PATH, cache_file=None):
    """Returns the dataset as a tuple of lists. The first list contains the data, the second the labels

    Args:
        dataset_name (str): the name of the dataset
        use_cached (bool, optional): whether to use the cached dataset
        dataset_folder (str, optional): where to search the dataset
        cache_path (str, optional): folder to save the dataset to after fetching it
        cache_file (str, optional): used to overwrite the cache-path

    Returns:
        tuple(list): two lists, the first the data, the second the labels
    """

    dataset_npy = cache_file if cache_file else os.path.join(cache_path, 'dataset_{}.npy'.format(dataset_name))

    if use_cached and os.path.exists(dataset_npy):
        X, Y = get_dataset_cached(dataset_npy)
    else:
        X, Y = get_dataset_module(dataset_folder, dataset_name).fetch()

        # Test dataset validity before saving it
        test_dataset_validity(X, Y)

        with open(dataset_npy, 'wb') as f:
            pickle.dump((X, Y), f)

    test_dataset_validity(X, Y)
    return X, Y


def get_dataset_cached(cache_file, check_validity=True):
    if not os.path.exists(cache_file):
        raise FileNotFoundError('Could not find cache_file: {}'.format(cache_file))

    with open(cache_file, 'rb') as f:
        res = pickle.load(f)
    if check_validity:
        X, Y = res
        test_dataset_validity(X, Y)
    return res


def get_concept_map_combined_dataset_for_dataset(dataset: str):
    concept_map_cache_file = get_all_cached_graph_datasets(dataset, graph_type=TYPE_CONCEPT_MAP)
    assert len(concept_map_cache_file) == 1
    concept_map_cache_file = concept_map_cache_file[0]
    X_combined, Y = graph_helper.get_combined_text_graph_dataset(concept_map_cache_file)
    assert len(X_combined) == len(Y)
    return X_combined, Y


def get_text_dataset_filtered_by_concept_map(dataset: str):
    X_combined, Y = get_concept_map_combined_dataset_for_dataset(dataset)
    X_text = [text for (_, text, _) in X_combined]
    assert isinstance(X_text[0], str)
    return X_text, Y


def get_all_available_dataset_names(dataset_folder=DATASET_FOLDER, limit_datasets=None):
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

    if limit_datasets is not None:
        dataset_folders = [x for x in dataset_folders if x in limit_datasets]

    return sorted(dataset_folders)


def get_all_cached_datasets(cache_path=CACHE_PATH):
    return sorted(glob(cache_path + '/*.npy'))


def get_all_cached_graph_datasets(dataset_name=None, graph_type=None, cache_path=CACHE_PATH):
    def graph_dataset_filter(cache_file):
        filename = filename_utils.get_filename_only(cache_file)
        is_graph_dataset = filename.startswith('dataset_graph')
        is_not_relabeled = '_relabeled' not in filename
        is_not_gram_or_phi = 'gram' not in filename and 'phi' not in filename
        is_in_dataset = not dataset_name or dataset_name == filename_utils.get_dataset_from_filename(filename)
        is_right_graph_type = not graph_type or graph_type == graph_helper.get_graph_type_from_filename(cache_file)

        return np.all([
            is_graph_dataset,
            is_not_relabeled,
            is_not_gram_or_phi,
            is_in_dataset,
            is_right_graph_type
        ])

    return [x for x in get_all_cached_datasets(cache_path) if graph_dataset_filter(x)]


def get_all_gram_datasets(dataset_name=None, cache_path=CACHE_PATH):
    gram_files = glob('{}/*gram*.npy'.format(cache_path))
    return [x for x in gram_files if not dataset_name or filename_utils.get_dataset_from_filename(x) == dataset_name]


def get_all_cached_graph_phi_datasets(dataset_name=None, cache_path=CACHE_PATH):
    if dataset_name:
        dataset_name = dataset_name.replace('-single', '')
    return [x for x in get_all_cached_datasets(cache_path) if 'phi' in x and (not dataset_name or filename_utils.get_dataset_from_filename(x) == dataset_name)]


def get_w2v_embedding_for_dataset(dataset_name, embedding_folder='data/embeddings/trained'):
    embedding_file = os.path.join(embedding_folder, dataset_name + '.npy')

    if not os.path.exists(embedding_file):
        raise FileNotFoundError('{}'.format(embedding_file))

    with open(embedding_file, 'rb') as f:
        return pickle.load(f)


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

    def import_from_path(path):
        return importlib.import_module(path.replace('/', '.'))

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


def get_gml_graph_dataset(dataset_name, use_cached=True, graphs_folder=GRAPHS_FOLDER, graph_type='concept_map', cache_npy: str = None, suffix: str = ''):
    """Retrieves the gml dataset.

    Args:
        dataset_name (str): the dataset
        use_cached (bool, optional):
        graphs_folder (str, optional): where to search the gml graphs

    Returns:
        tuple(list, list): X and Y
    """
    from utils import graph_helper

    if graphs_folder in dataset_name:
        dataset_name = dataset_name.replace(graphs_folder + '/', '')

    graph_folder = os.path.join(graphs_folder, dataset_name)

    if not cache_npy:
        cache_npy = os.path.join(CACHE_PATH, 'dataset_graph_{}_{}{}.npy'.format(graph_type, dataset_name, suffix))

    if use_cached and os.path.exists(cache_npy):
        with open(cache_npy, 'rb') as f:
            X, Y = pickle.load(f)
    else:
        X, Y = graph_helper.get_graphs_from_folder(graph_folder, undirected=False)
        # Test dataset validity before saving it
        test_dataset_validity(X, Y)

        with open(cache_npy, 'wb') as f:
            pickle.dump((X, Y), f)
    return X, Y


def get_subset_with_most_frequent_classes(X, Y, num_classes_to_keep=2, use_numpy=False):
    most_common_classes = [label for label, count in Counter(Y).most_common(num_classes_to_keep)]
    if use_numpy:
        # A little slower
        indices = np.in1d(Y, most_common_classes)
        return np.array(X, dtype=str)[indices], np.array(Y, dtype=str)[indices]
    else:
        most_common_classes = set(most_common_classes)
        data = [(text, label) for text, label in zip(X, Y) if label in most_common_classes]
        return [x[0] for x in data], [x[1] for x in data]


def get_dataset_subset_with_most_frequent_classes(dataset_name: str, num_classes_to_keep: int = 2, get_dataset_kwargs: dict = None) -> tuple:
    """Given a base dataset, this function returns the num_classes_to_keep classes.

    Args:
        dataset_name (str): base dataset
        num_classes_to_keep (int, optional): how many classes should be taken

    Returns:
        tuple(list, list): X and Y, where the dataset only contains data from the most frequent classes
    """
    X, Y = get_dataset(dataset_name, **(get_dataset_kwargs or {}))
    return get_subset_with_most_frequent_classes(X, Y, num_classes_to_keep=num_classes_to_keep)


def test_dataset_validity(X, Y):
    def sparse_matrix_type(a):
        return isinstance(a, (scipy.sparse.lil_matrix, scipy.sparse.coo_matrix))

    assert isinstance(X, list) or sparse_matrix_type(X), 'X must be a list, it is: {}'.format(type(X))
    assert isinstance(Y, list), 'Y must be a list'
    if sparse_matrix_type(X):
        assert X.shape[0] and len(Y), 'Dataset is empty'
        assert X.shape[0] == len(Y), 'X and Y must have same length'
    else:
        assert len(X) and len(Y), 'Dataset is empty'
        assert len(X) == len(Y), 'X and Y must have same length'
    assert len(set(Y)), 'Y must contain at least one label'