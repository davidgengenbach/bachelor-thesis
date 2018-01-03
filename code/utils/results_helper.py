import collections
from utils import helper, filename_utils, git_utils, time_utils, significance_test_utils
from utils.remove_coefs_from_results import remove_coefs_from_results
import pandas as pd
from glob import glob
import os
import pickle
import re
import typing
import numpy as np
import sklearn
import typing
import shutil

from utils.logger import LOGGER
import tqdm

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


replacements = [
    ('param_', ''),
    ('features__fast_wl_pipeline__feature_extraction__feature_extraction__', 'graph__'),
    ('features__fast_wl_pipeline__feature_extraction__', ''),
    ('graph_preprocessing', 'graph__preprocessing'),
    #('normalizer', ''),
    ('preprocessing', 'text__preprocessing'),
    ('features__fast_wl_pipeline__feature_extraction__normalizer', 'graph__normalizer'),
    ('feature_extraction__fast_wl__', 'graph__fast_wl__'),
    ('feature_extraction__phi_picker__', 'graph__phi_picker__'),
    ('features__text__vectorizer__', 'text__'),
    ('graph_to_text__use_edges', 'graph__graph_to_text__use_edges'),
    ('vectorizer', 'text__vectorizer')
]


def clean_result_keys(el: dict) -> dict:
    results = {}
    for key, val in el.items():
        for search, replace in replacements:
            if key.startswith(search):
                key = key.replace(search, replace)
        results[key] = val
    return results


def get_kernel_from_filename(filename: str) -> str:
    for kernel in ['spgk', 'wl', 'tfidf']:
        if kernel in filename:
            return kernel

    parts = []
    if '.simple.' in filename:
        parts.append('simple_set_matching')
    if 'graph_text_' in filename or 'graph_content_only' in filename:
        parts.append('text')
    if 'combined' in filename or '__graph__' in filename or 'graph_structure_only' in filename or 'graph_relabel' in filename:
        parts.append('wl')
    if 'node_weight' in filename:
        parts.append('wl_nodeweight')

    if not len(parts):
        raise Exception('Unknown kernel: {}'.format(filename))

    return '_'.join(parts)


def _get_experiment_name_from_filename(x):
    filename = filename_utils.get_filename_only(x, with_extension=False)
    parts = x.split('__')
    if 'result___experiment' not in x or len(parts) < 4: return ''
    return parts[1][1:].strip()


def get_results(folder=None, use_already_loaded=False, results_directory=RESULTS_DIR, log_progress=tqdm.tqdm_notebook, exclude_filter=None, include_filter=None, filter_out_non_complete_datasets=None  , remove_split_cols=True, remove_rank_cols=True, remove_fit_time_cols=True, filter_out_experiment=None, ignore_experiments=True, only_load_dataset=None, fetch_predictions=False):
    '''This function _seriously_ has to be refactored!'''
    global _DF_ALL, _RESULT_CACHE

    if not use_already_loaded:
        _RESULT_CACHE = []
        _DF_ALL = None

    result_folders = get_result_folders(results_directory)

    folder = 'data/results/{}'.format(folder) if folder else result_folders[-1]

    result_files = get_result_filenames_from_folder(folder)
    if filter_out_experiment:
        result_files = [x for x in result_files if _get_experiment_name_from_filename(x) == filter_out_experiment]

    if ignore_experiments and not filter_out_experiment:
        result_files = [x for x in result_files if 'experiment_' not in x]

    if only_load_dataset is not None:
        result_files = [x for x in result_files if filename_utils.get_dataset_from_filename(x) in only_load_dataset]

    data_ = []
    for result_file in log_progress(result_files) if log_progress else result_files:
        if include_filter and include_filter not in result_file: continue
        if exclude_filter and exclude_filter in result_file: continue
        if '_nested_' in result_file: continue

        dataset_name = filename_utils.get_dataset_from_filename(result_file)

        if result_file in _RESULT_CACHE:
            continue


        _RESULT_CACHE.append(result_file)

        with open(result_file, 'rb') as f:
            result_data = pickle.load(f)

        remove_transformer_classes(result_data)

        result_file = result_file.split('/')[-1]
        info = {}
        if 'results' in result_data:
            info = {'info__' + k: v for k, v in result_data.get('meta_data', result_data).items() if k != 'results'}
        result = result_data if 'params' in result_data else result_data['results']
        assert 'params' in result

        result = clean_result_keys(result)
        for idx, el in enumerate(result['params']):
            result['params'][idx] = clean_result_keys(el)

        prediction_file = '{}/predictions/{}'.format(folder, filename_utils.get_filename_only(result_file))
        predictions_exist = os.path.exists(prediction_file)

        num_results = len(result['params'])
        result['prediction_file_exists'] = [predictions_exist] * num_results

        if fetch_predictions and not predictions_exist:
            LOGGER.warning('fetch_predictions=True but could not find prediction: {}'.format(prediction_file))

        if fetch_predictions and predictions_exist:
            with open(prediction_file, 'rb') as f:
                r = pickle.load(f)
            result_git_commit = result_data['meta_data']['git_commit']
            git_commit = r['meta_data']['git_commit']
            if not git_commit == result_git_commit:
                LOGGER.warning('Unmatching git commit for prediction/result file! Prediction: {}, Result: {}'.format(git_commit, result_git_commit))
            else:
                prediction = r['results']
                Y_real, Y_pred, X_test = prediction['Y_real'], prediction['Y_pred'], prediction['X_test']
                scores = get_scores(Y_real, Y_pred)
                for name, val in scores.items():
                    result['prediction_score_{}'.format(name)] = [val] * num_results
                result['prediction_file'] = [prediction_file] * num_results


        is_graph_dataset = '_graph__dataset' in result_file or 'graph_combined__dataset' in result_file or '__graph_node_weights__dataset_' in result_file or 'graph_cooccurrence' in result_file or '__graph_structure_only__' in result_file or '_graph_relabel' in result_file or '__graph_content_only__' in result_file
        result['combined'] = 'graph_combined__dataset_' in result_file
        result['kernel'] = 'unknown'
        if is_graph_dataset:
            is_cooccurrence_dataset = 'cooccurrence' in result_file

            result['type'] = 'cooccurrence' if is_cooccurrence_dataset else 'concept_map'
            result['lemmatized'] = '_lemmatized_' in result_file

            result['kernel'] = get_kernel_from_filename(result_file)
            if 'graph__fast_wl__node_weight_function' in result:
                result['graph__fast_wl__node_weight_function'] = ['none' if x is None else x.__name__ for x in result.get('graph__fast_wl__node_weight_function')]

            # Co-Occurrence
            if is_cooccurrence_dataset:
                parts = re.findall(r'cooccurrence_(.+?)_(.+?)_', result_file)[0]
                result['window_size'], result['words'] = parts
            # Concept Maps
            else:
                result['words'] = 'concepts'
        elif 'dummy' in result_file:
            result['type'] = 'dummy'
            result['words'] = 'dummy'
        else:
            result['type'] = 'text'
            result['words'] = ['all' for x in result['params']]

        if dataset_name.endswith('-single') or dataset_name.endswith('-ana'):
            dataset_name = dataset_name.rsplit('-', 1)[0]

        if 'time_checkpoints' in result_data:
            timestamps = result_data['time_checkpoints']
            timestamps = sorted(timestamps.items(), key=lambda x: x[1])

            start = timestamps[0][1]
            end = timestamps[-1][1]

            result['timestamps'] = [timestamps] * num_results
            result['time'] = [end - start] * num_results

        result['filename'] = result_file
        result['dataset'] = dataset_name

        for k, v in info.items():
            result[k] = [v] * len(result['params'])
        data_.append(result)

    for d in data_:
        result_df = pd.DataFrame(d)
        _DF_ALL = result_df if _DF_ALL is None else _DF_ALL.append(result_df)

    if _DF_ALL is None or not len(_DF_ALL):
        LOGGER.warning('Did not retrieve results! Aborting')
        return None

    if filter_out_non_complete_datasets:
        # Only keep datasets where there are all three types (text, co-occurrence and concept-graph) of results
        df_all = filter_out_datasets(_DF_ALL, lambda x: len(x.type.value_counts()) == filter_out_non_complete_datasets)
    else:
        df_all = _DF_ALL

    # Remove cols
    df_all = df_all[[
        x for x in df_all.columns.tolist() if
        (not remove_split_cols or not re.match(r'^split\d', x)) and
        (not remove_fit_time_cols or not re.match(r'_time$', x)) and
        (not remove_rank_cols or not re.match(r'rank_', x))
    ]]

    prio_columns = ['dataset', 'type', 'combined']
    low_prio_columns = ['params', 'filename'] + [c for c in df_all.columns if c.startswith('std_') or c.startswith('mean_')]
    columns = df_all.columns.tolist()
    for c in prio_columns + low_prio_columns:
        columns.remove(c)

    # df_all = df_all.fillna(value = 'none')

    return df_all.reset_index(drop=True)[prio_columns + columns + low_prio_columns]  # .set_index('filename')

def get_scores(Y_true, Y_pred, metrics=significance_test_utils.metrics):
    results = {}
    for metric_name, fn in metrics:
        results[metric_name] = fn(Y_true, Y_pred)
    return results


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


def get_result_filenames_from_folder(folder: str=None):
    if not folder:
        folder = get_result_folders()[-1]
    return glob('{}/*.npy'.format(folder))


def get_result_for_prediction(prediction_filename):
    results_filename = prediction_filename.replace('/predictions', '')
    if not os.path.exists(results_filename): return None

    with open(results_filename, 'rb') as f:
        return pickle.load(f)

def get_prediction(filename):
    print(filename)



def get_predictions_files(folder: str = None):
    if not folder:
        folder = get_result_folders()[-1]

    prediction_folder = '{}/predictions'.format(folder)
    return glob('{}/*.npy'.format(prediction_folder))


def get_predictions(folder: str = None, filenames: typing.Collection = None) -> typing.Generator:
    '''
    Returns the predictions (filename: str, prediction: dict) for a given result folder (or the most frequent, if the given folder is none).


    Args:
        folder: the result folder to be considered. If None is given, the most recent folder is used.
        filenames: a filter for considered filenames. If None is given, all files are returned.

    Returns:
        A generator of tuples (filename, prediction)
    '''
    for prediction_file in get_predictions_files(folder):
        p_filename = prediction_file.split('/')[-1]
        if filenames and p_filename not in filenames: continue

        with open(prediction_file, 'rb') as f:
            prediction = pickle.load(f)
        yield prediction_file, prediction


def cleanup_outdated_predictions(results_folder: str=None, dry_run=True, ignore_filter: str = 'dummy'):
    folder = 'data/results/{}'.format(results_folder) if results_folder else get_result_folders()[-1]
    result_files = get_result_filenames_from_folder(folder)

    for result_file in result_files:
        if ignore_filter and ignore_filter in result_file: continue

        prediction_file = '{}/predictions/{}'.format(folder, filename_utils.get_filename_only(result_file))
        predictions_exist = os.path.exists(prediction_file)

        if not predictions_exist:
            LOGGER.warning('Did not find prediction file for: {}'.format(result_file))
            continue

        with open(result_file, 'rb') as f:
            result_data = pickle.load(f)
        with open(prediction_file, 'rb') as f:
            r = pickle.load(f)

        result_git_commit = result_data['meta_data']['git_commit']
        git_commit = r['meta_data']['git_commit']
        if result_git_commit != git_commit:
            if dry_run:
                LOGGER.info('Outdated prediction: {}. dry_run=True, so it will not get deleted'.format(prediction_file))
                continue
            LOGGER.info('Outdated prediction: {}. Deleting'.format(prediction_file))
            os.remove(prediction_file)


def filter_out_datasets(df, fn):
    return df.groupby('dataset').filter(fn).reset_index(drop=True)


def get_experiments_by_names(names: list, fill_na='-', **get_results_kwargs) -> pd.DataFrame:
    df = pd.DataFrame()
    for x in names:
        df_ = get_results(filter_out_experiment=x, filter_out_non_complete_datasets=False, **get_results_kwargs)
        df = df.append(df_)
    if fill_na:
        df = df.fillna(fill_na)
    return df.reset_index()


def save_results(gscv_result, filename, info=None, remove_coefs=True, time_checkpoints=None):
    if remove_coefs:
        remove_coefs_from_results(gscv_result)

    dump_pickle_file(info, filename, dict(
        results=gscv_result,
        time_checkpoints=time_checkpoints
    ))




def dump_pickle_file(args, filename: str, data: dict, add_meta: bool = True):
    meta = dict(meta_data=get_metadata(args)) if add_meta else dict()

    with open(filename, 'wb') as f:
        pickle.dump(dict(meta, **data), f)


def get_metadata(args, other=None) -> dict:
    return dict(
            git_commit=str(git_utils.get_current_commit()),
            timestamp=time_utils.get_timestamp(),
            timestamp_readable=time_utils.get_time_formatted(),
            args=args,
            other=other
    )