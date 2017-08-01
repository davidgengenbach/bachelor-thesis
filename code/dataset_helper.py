from glob import glob
import os
import importlib
import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import pickle

PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = 'data/datasets'

def split_dataset(dataset_dict, train_size = 0.8, random_state_for_shuffle = 42, max_elements = -1):
    train_set = []
    test_set = []
    for topic, docs in dataset_dict.items():
        topic_id = topic
        # Create train/test split
        if train_size == 1.0:
            docs_train, docs_test = shuffle(
                docs, random_state=random_state_for_shuffle
            ), []
        else:
            docs_train, docs_test = train_test_split(
                docs,
                train_size=train_size,
                random_state=random_state_for_shuffle
            )
        if max_elements != -1:
            max_elements_train = min(int(max_elements * train_size), len(docs_train))
            max_elements_test = min(int(max_elements * (1 - train_size)), len(docs_test))
            docs_train = docs_train[:max_elements_train]
            docs_test = docs_test[:max_elements_test]

        assert len(docs_train) > 0, "\t-> len(docs_train) == 0"
        assert train_size == 1.0 or len(docs_test) > 0, "\t-> len(docs_test) == 0"
        train_set += docs_train
        test_set += docs_test
    return train_set, test_set


def get_dataset_dict_preprocessed(dataset_name, dataset_folder = DATASET_FOLDER, use_cached = True):
    X, Y = get_dataset(dataset_name, dataset_folder = dataset_folder, use_cached = use_cached)
    X = [preprocessing.preprocess_text(text) for text in X]
    return X, Y, set(Y)

def get_dataset(dataset_name, use_cached = True, dataset_folder=DATASET_FOLDER):
    """Returns the dataset as a list of docs with labels: [(topic1, document1], (topic2, document2))]
    
    Args:
        dataset_name (str): The name of the dataset
        use_cached (bool, optional): Whether to use the cached dataset
        dataset_folder (str, optional): Where to search the dataset
    
    Returns:
        list(tuples): a list of documents with labels
    """
    dataset_folder_ = os.path.join(dataset_folder, dataset_name, 'dataset')
    assert os.path.exists(
        dataset_folder_ + '.py'), 'dataset.py for dataset "{}" does not exist! ({})'.format(dataset_name, dataset_folder_ + '.py')

    dataset_npy = os.path.join(*(dataset_folder_.split('/')[:-1] + ['dataset.npy']))

    if use_cached and os.path.exists(dataset_npy):
        with open(dataset_npy, 'rb') as f:
            return pickle.load(f)

    dataset_module = importlib.import_module(dataset_folder_.replace('/', '.'))
    assert hasattr(dataset_module, 'fetch'), 'dataset {} does not have a fetch method'.format(dataset_name)
    X, Y = dataset_module.fetch()
    assert isinstance(X, list), 'X must be a list'
    assert isinstance(Y, list), 'Y must be a list'
    assert len(X) and len(Y), 'Dataset is empty'
    assert len(X) == len(Y), 'X and Y must have same length'

    with open(dataset_npy, 'wb') as f:
        pickle.dump((X, Y), f)

    return X, Y


def get_all_datasets_raw(dataset_dir=DATASET_FOLDER):
    """Returns a dict with the available datasets as key and the documents as values

    Args:
        dataset_dir (str, optional): Where to search for datasets

    Returns:
        dict: Keys are the dataset names, the values is a list of docs like [(topic1, document1], (topic2, document2))]
    """
    datasets = glob('{}/*/dataset.py'.format(dataset_dir))
    dataset_folders = [x.replace('/dataset.py', '').replace('/', '.') for x in datasets]
    return {x.split('.')[-1]: get_dataset(x.split('.')[-1], dataset_dir) for x in dataset_folders}

def get_dataset_dict(X, Y = None):
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

if __name__ == '__main__':
    get_all_datasets_raw()
