from utils import dataset_helper
import os

all_datasets = dataset_helper.get_all_available_dataset_names()

def get_dataset_from_filename(filename: str, ignore_subtype = False) -> str:
    filename = filename.split('/')[-1]
    candidates = sorted([dataset for dataset in all_datasets if dataset in filename])
    assert len(candidates) <= 2
    dataset = None

    if len(candidates):
        dataset = sorted(candidates)[0 if ignore_subtype else -1]

    return dataset


def get_abs_path(file):
    return os.path.abspath('./' + file)

def get_filename_only(file):
    return file.rsplit('/', 1)[-1]

def remove_file_extension(file):
    return file.rsplit('.', 1)[0]


def get_result_filename_for_task(task):
    dataset = get_dataset_from_filename(task.name, ignore_subtype=False)

    parts = [
        'result_',
        dataset,
        task.type
    ]

    file = remove_file_extension(task.name)
    if dataset != file:
        parts.append(file)

    return '__'.join(parts) + '.npy'
