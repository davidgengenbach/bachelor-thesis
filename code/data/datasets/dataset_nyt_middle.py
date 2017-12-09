from utils import dataset_helper

def fetch(dataset='nyt', num_per_class=80):
    X, Y = dataset_helper.get_dataset(dataset)
    return dataset_helper.get_num_elements_per_class(X, Y, num_per_class=num_per_class)