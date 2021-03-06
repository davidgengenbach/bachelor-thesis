from utils import dataset_helper


def fetch(base_set='reuters-21578'):
    X, Y = dataset_helper.get_dataset_subset_with_most_frequent_classes(base_set, num_classes_to_keep=8)
    return X, Y
