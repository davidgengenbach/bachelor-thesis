from glob import glob
import os
import importlib

def get_dataset(dataset_name, dataset_folder = 'data/datasets'):
    """Returns the dataset as a list of docs with labels: [(topic1, document1], (topic2, document2))]
    
    Args:
        dataset_name (TYPE): The name of the dataset
        dataset_folder (str, optional): Where to search the dataset
    
    Returns:
        list(tuples): a list of documents with labels
    """
    dataset_py = os.path.join(dataset_folder, dataset_name, 'dataset')
    assert os.path.exists(dataset_py + '.py'), 'dataset.py for dataset "{}" does not exist! ({})'.format(dataset_name, dataset_py + '.py')
    dataset_module = importlib.import_module(dataset_py.replace('/', '.'))
    assert hasattr(dataset_module, 'fetch'), 'dataset {} does not have a fetch method'.format(dataset_name)
    data = dataset_module.fetch()
    assert isinstance(data, list), 'data must be a list of tuples'
    assert len(data), 'Dataset is empty'
    return data

def get_all_datasets_raw(dataset_dir = 'data/datasets'):
    """Returns a dict with the available datasets as key and the documents as values
    
    Args:
        dataset_dir (str, optional): Where to search for datasets
    
    Returns:
        dict: Keys are the dataset names, the values is a list of docs like [(topic1, document1], (topic2, document2))]
    """
    datasets = glob('{}/*/dataset.py'.format(dataset_dir))
    dataset_folders = [x.replace('/dataset.py', '').replace('/', '.') for x in datasets]
    return { x.split('.')[-1]: get_dataset(x.split('.')[-1], dataset_dir) for x in dataset_folders }

if __name__ == '__main__': get_all_datasets_raw()