import os
from utils import dataset_helper, classification_options
import re


def get_dataset_from_filename(filename: str, ignore_subtype = False) -> str:
    all_datasets = dataset_helper.get_all_available_dataset_names()

    filename = filename.split('/')[-1]
    candidates = sorted([dataset for dataset in all_datasets if dataset in filename])
    assert len(candidates) <= 3
    dataset = None

    if len(candidates):
        dataset = sorted(candidates, key=lambda x: len(x))[0 if ignore_subtype else -1]

    return dataset


def get_abs_path(file):
    return os.path.abspath('./' + file)

def get_filename_only(file, with_extension = True):
    filename = file.rsplit('/', 1)[-1]
    if not with_extension:
        filename = filename.rsplit('.', 1)[0]
    return filename

def remove_file_extension(file):
    return file.rsplit('.', 1)[0]

_COOCCURRENCE_REGEXP = r'_cooccurrence_(\d)_(.+?)_'

def get_cooccurrence_window_size_from_filename(filename):
    match = re.findall(_COOCCURRENCE_REGEXP, filename)
    assert len(match) == 1
    return int(match[0][0])

def get_cooccurrence_words_from_filename(filename):
    match = re.findall(_COOCCURRENCE_REGEXP, filename)
    assert len(match) == 1
    return match[0][1]


def get_result_filename_for_task(task, experiment_config:dict=None, cfo: classification_options.ClassificationOptions = None):
    dataset = get_dataset_from_filename(task.name, ignore_subtype=False)
    file = remove_file_extension(get_filename_only(task.name))
    experiment_name = experiment_config.get('experiment_name', None) if experiment_config else None

    parts = [
        'result_',
        dataset,
        task.type
    ]

    if experiment_name:
        parts.insert(1, experiment_name)

    if cfo.use_nested_cross_validation:
        parts.insert(-1, 'nested')

    if dataset != file:
        parts.append(file)

    return '__'.join(parts) + '.npy'


def get_topn_threshold_from_lookupfilename(x):
    matches = re.findall(r'threshold-(.+?)\.topn-(.+?)\.', x)
    assert len(matches) == 1
    return matches[0]