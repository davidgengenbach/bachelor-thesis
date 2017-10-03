import collections
from utils import helper
import pandas as pd
from glob import glob
import os
import pickle

_RESULT_CACHE = []
_DF_ALL = None

RESULTS_DIR = 'data/results'

def get_results(folder=None, use_already_loaded=True, results_directory=RESULTS_DIR, log_progress=True, exclude_filter = None, filter_out_non_complete_datasets = False):
    global _DF_ALL, _RESULT_CACHE

    if not use_already_loaded:
        _RESULT_CACHE = []
        _DF_ALL = None

    result_folders = get_result_folders(results_directory)

    folder = 'data/results/{}'.format(folder) if folder else result_folders[-1]

    cache_counter = collections.Counter()

    # Get all result files (pickled)
    result_files = get_result_filenames_from_folder(folder)

    for result_file in helper.log_progress(result_files) if log_progress else result_files:
        if exclude_filter and exclude_filter in result_file: continue
        filename = result_file.split('/')[-1]
        if filename in _RESULT_CACHE:
            cache_counter['cached'] += 1
            continue

        cache_counter['loaded'] += 1
        _RESULT_CACHE.append(filename)

        with open(result_file, 'rb') as f:
            result = pickle.load(f)

        dataset = result_file.split('/')[-1].rsplit('.', 2)[0]
        is_graph_dataset = 'graph' in dataset
        is_cooccurrence_dataset = 'cooccurrence' in dataset
        dataset_name = dataset

        result['combined'] = 'combined' in result_file
        if is_graph_dataset:
            is_lemmatized = '_lemmatized_' in result_file
            result['lemmatized'] = is_lemmatized

            result['same_label'] = 'same-label' in result_file
            is_simple_kernel = '.simple.' in result_file
            if is_simple_kernel:
                result['kernel'] = 'simple_set_matching'
            else:
                result['kernel'] = 'spgk' if 'spgk' in result_file else 'wl'
            is_relabeled = 'relabeled' in result_file
            result['relabeled'] = is_relabeled
            if is_relabeled:
                topn = result_file.split('topn-')[1].split('_')[0]
                threshold = result_file.split('threshold-')[1].split('_')[0]
                result['topn'] = int(topn)
                result['threshold'] = float(threshold)
            if result['kernel'] == 'wl':
                result['wl_iteration'] = dataset.split('.')[-1]
            parts = dataset.split('_')
            if is_cooccurrence_dataset:
                dataset_name = parts[-1].split('_')[0].split('.')[0]
                result['words'] = parts[4]
                result['window_size'] = parts[3]
            # GML
            else:
                dataset_name = parts[-1].split('.')[0]
                result['words'] = 'concepts'
            result['type'] = 'cooccurrence' if is_cooccurrence_dataset else 'concept-graph'
        else:
            result['type'] = 'text'
            dataset_name = dataset.split('_')[1]
            result['words'] = ['all' for x in result['params']]

        result['classifier'] = [None] * len(result['params'])
        for idx, param in enumerate(result['params']):
            result['classifier'][idx] = type(param['clf']).__name__
            del param['clf']
        del result['param_clf']

        dataset_name = dataset_name.replace('-single', '').replace('-ana', '').strip()

        if '-ana' in result_file:
            result['is_ana'] = True

        if dataset_name.endswith('-single') or dataset_name.endswith('-ana'):
            dataset_name = dataset_name.rsplit('-', 1)[0]

        result['filename'] = result_file
        result['dataset'] = dataset_name

        result_df = pd.DataFrame(result)
        _DF_ALL = result_df if _DF_ALL is None else _DF_ALL.append(result_df)

    assert _DF_ALL is not None
    assert len(_DF_ALL)

    # Only keep datasets where there are all three types (text, co-occurence and concept-graph) of results

    if filter_out_non_complete_datasets:
        df_all = _DF_ALL.groupby('dataset').filter(lambda x: len(x.type.value_counts()) == 3).reset_index(drop=True)
    else:
        df_all = _DF_ALL

    return df_all


def get_result_folder_df(results_directory=RESULTS_DIR):
    """Collects all results folders and the number of result files in them and returns them as a pandas.DataFrame.

    Args:
        results_directory (str, optional): where the result folders are located

    Returns:
        pandas.DataFrame: the result folder with the number of results in them.
    """
    result_folders = get_result_folders(results_directory)
    return pd.DataFrame([(folder.split('/')[-1], len(glob('{}/*.npy'.format(folder)))) for folder in result_folders], columns=['result_folder', 'num_results']).set_index('result_folder').sort_index()


def get_result_folders(results_directory=RESULTS_DIR):
    return [x for x in glob('{}/*'.format(results_directory)) if os.path.isdir(x) and len(glob('{}/*.npy'.format(x)))]


def get_result_filenames_from_folder(folder):
    return glob('{}/*.npy'.format(folder))