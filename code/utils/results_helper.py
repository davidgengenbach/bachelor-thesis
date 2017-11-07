import collections
from utils import helper, filename_utils
import pandas as pd
from glob import glob
import os
import pickle
import re
import typing
import numpy as np
import sklearn
import typing

_RESULT_CACHE = []
_DF_ALL = None

RESULTS_DIR = 'data/results'

def remove_transformer_classes(d):
    def get_typename(val):
        return type(val).__name__

    if isinstance(d, dict):
        for key, val in d.items():
            if val is None: continue
            if isinstance(val, (np.ma.masked_array, list)):
                for idx, x in enumerate(val):
                    if isinstance(x, sklearn.base.BaseEstimator):
                        val[idx] = get_typename(x)
                    if isinstance(x, dict):
                        remove_transformer_classes(x)
            if isinstance(val, sklearn.base.BaseEstimator):
                d[key] = get_typename(val)
            if isinstance(val, dict):
                remove_transformer_classes(val)


def get_results(folder=None, use_already_loaded=True, results_directory=RESULTS_DIR, log_progress=True, exclude_filter = None, filter_out_non_complete_datasets = True, remove_split_cols = True, remove_rank_cols = True, remove_fit_time_cols = True):
    global _DF_ALL, _RESULT_CACHE

    if not use_already_loaded:
        _RESULT_CACHE = []
        _DF_ALL = None

    result_folders = get_result_folders(results_directory)

    folder = 'data/results/{}'.format(folder) if folder else result_folders[-1]

    # Get all result files (pickled)
    result_files = get_result_filenames_from_folder(folder)

    data_ = []
    for result_file in helper.log_progress(result_files) if log_progress else result_files:
        if exclude_filter and exclude_filter in result_file: continue

        dataset_name = filename_utils.get_dataset_from_filename(result_file)

        if result_file in _RESULT_CACHE:
            continue

        _RESULT_CACHE.append(result_file)

        with open(result_file, 'rb') as f:
            result_data = pickle.load(f)

        remove_transformer_classes(result_data)

        result_file = result_file.split('/')[-1]

        result = result_data if 'params' in result_data else result_data['results']

        is_graph_dataset = '_graph_' in result_file
        result['combined'] = 'combined' in result_file

        result['kernel'] = 'unknown'

        if is_graph_dataset:
            is_cooccurrence_dataset = 'cooccurrence' in result_file

            result['type'] = 'cooccurrence' if is_cooccurrence_dataset else 'concept-graph'
            result['lemmatized'] = '_lemmatized_' in result_file
            result['same_label'] = 'same_label' in result_file

            is_tfidf = 'tfidf' in result_file
            result['is_tfidf'] = is_tfidf

            is_simple_kernel = '.simple.' in result_file


            if is_simple_kernel:
                result['kernel'] = 'simple_set_matching'
            elif 'spgk' in result_file:
                result['kernel'] = 'spgk'
            elif 'wl' in result_file:
                result['kernel'] = 'wl'
            elif 'tfidf' in result_file:
                result['kernel'] = 'tfidf'
            elif '_graph_text_' in result_file:
                result['kernel'] = 'text'
            else:
                assert False

            result['wl_return_iteration'] = [None] * len(result['params'])
            result['wl_round_to_decimals'] = [None] * len(result['params'])

            for idx, param in enumerate(result['params']):
                if param is None: continue
                for k, v in param.items():
                    for x in ['return_iteration', 'round_to_decimals']:
                        if k.endswith(x):
                            result['wl_' + x][idx] = v


            if result['kernel'] == 'wl':
                # ....
                wl_iterations = [[val for key, val in val.items() if key.endswith('return_iteration')][0] for val in result['params']]
                result['wl_iteration'] = wl_iterations


            is_relabeled = 'relabeled' in result_file
            result['relabeled'] = is_relabeled
            if is_relabeled:
                topn = result_file.split('topn-')[1].split('_')[0]
                threshold = result_file.split('threshold-')[1].split('_')[0]
                result['topn'] = int(topn)
                result['threshold'] = float(threshold)

            # Co-Occurrence
            if is_cooccurrence_dataset:
                parts = re.findall(r'cooccurrence_(.+?)_(.+?)_', result_file)[0]
                result['window_size'], result['words'] = parts
            # Concept Maps
            else:
                result['words'] = 'concepts'
        else:
            result['type'] = 'text'
            result['words'] = ['all' for x in result['params']]

        result['classifier'] = [x['classifier'] if 'classifier' in x else None for x in result['params']]
        result['is_ana'] = '-ana' in result_file

        if dataset_name.endswith('-single') or dataset_name.endswith('-ana'):
            dataset_name = dataset_name.rsplit('-', 1)[0]

        result['filename'] = result_file
        result['dataset'] = dataset_name

        data_.append(result)

    for d in data_:
        result_df = pd.DataFrame(d)
        _DF_ALL = result_df if _DF_ALL is None else _DF_ALL.append(result_df)

    assert _DF_ALL is not None
    assert len(_DF_ALL)

    if filter_out_non_complete_datasets:
        # Only keep datasets where there are all three types (text, co-occurrence and concept-graph) of results
        df_all = _DF_ALL.groupby('dataset').filter(lambda x: len(x.type.value_counts()) == 3).reset_index(drop=True)
    else:
        df_all = _DF_ALL

    # Remove cols
    df_all = df_all[[
        x for x in df_all.columns.tolist() if
            (not remove_split_cols or not re.match(r'^split\d', x)) and
            (not remove_fit_time_cols or not re.match(r'_time$', x)) and
            (not remove_rank_cols or not re.match(r'rank_', x))
    ]]

    return df_all.reset_index(drop=True)


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
    return sorted([x for x in glob('{}/201*'.format(results_directory)) if os.path.isdir(x) and len(glob('{}/*.npy'.format(x)))])


def get_result_filenames_from_folder(folder):
    return glob('{}/*.npy'.format(folder))

def get_result_for_prediction(prediction_filename):
    results_filename = prediction_filename.replace('/predictions', '')
    if not os.path.exists(results_filename): return None

    with open(results_filename, 'rb') as f:
        return pickle.load(f)

def get_predictions(folder: str=None) -> typing.Generator:
    if not folder:
        folder = get_result_folders()[-1]

    prediction_folder = '{}/predictions'.format(folder)

    for prediction_file in glob('{}/*.npy'.format(prediction_folder)):
        with open(prediction_file, 'rb') as f:
            prediction = pickle.load(f)
        yield prediction_file, prediction
