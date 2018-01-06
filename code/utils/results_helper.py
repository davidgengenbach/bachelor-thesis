import collections
from utils import helper, filename_utils, git_utils, time_utils, significance_test_utils, constants
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

from utils.logger import LOGGER
import tqdm

def get_results(
        folder=None,
        results_directory=constants.RESULTS_FOLDER,
        log_progress=tqdm.tqdm_notebook,
        exclude_filter=None,
        include_filter=None,
        remove_split_cols=True,
        remove_rank_cols=True,
        remove_fit_time_cols=True,
        filter_out_experiment=None,
        ignore_experiments=True,
        only_load_dataset=None,
        fetch_predictions=False
):
    '''
    Retrieves results from result folder.

    Note: This function _seriously_ has to be refactored!

    Args:
        folder: specify the results folder. If not specified, defaults to the most recent results folder
        results_directory: the base folder
        log_progress: function to log the progess. Takes an iterable and yields the item
        exclude_filter: which files to exclude
        include_filter: which files to include
        remove_split_cols: whether to keep the individual results for each split in CV
        remove_rank_cols: whether to keep the rank information in the CV results
        remove_fit_time_cols: keep the fit time
        filter_out_experiment: string thats gets filtered out
        ignore_experiments:
        only_load_dataset: filter the dataset
        fetch_predictions: whether to also retrieve the predictions and calculate the results on them

    Returns:
        pd.DataFrame: the results
    '''
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

        if '_nested_' in result_file:
            LOGGER.warning('Encountered nested CV result file. Currently not implemented. File: {}'.format(result_file))
            continue

        dataset_name = filename_utils.get_dataset_from_filename(result_file)

        with open(result_file, 'rb') as f:
            result_data = pickle.load(f)

        remove_transformer_classes(result_data)

        result_file = filename_utils.get_filename_only(result_file)
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

        # Fetch predictions and check whether the git commits are the same.
        # Also, calculate the prediction scores
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
                scores = calculate_scores(Y_real, Y_pred)
                for name, val in scores.items():
                    result['prediction_score_{}'.format(name)] = [val] * num_results
                result['prediction_file'] = [prediction_file] * num_results

        def is_graph_dataset():
            graph_file_types = [
                constants.TYPE_CONCEPT_MAP,
                constants.TYPE_COOCCURRENCE,
                'graph_extra'
            ]
            is_graph_dataset_ = False
            for x in graph_file_types:
                if '_{}_'.format(x) in result_file:
                    is_graph_dataset_ = True
                    break
            return is_graph_dataset_

        result['combined'] = np.any([
            'graph_combined__dataset_' in result_file,
            'graph_text_combined__dataset_' in result_file
        ])

        # TEXT
        if is_graph_dataset():
            is_cooccurrence_dataset = constants.TYPE_COOCCURRENCE in result_file
            result['type'] = constants.TYPE_COOCCURRENCE if is_cooccurrence_dataset else constants.TYPE_CONCEPT_MAP

            result['lemmatized'] = '_lemmatized_' in result_file
            result['kernel'] = get_kernel_from_filename(result_file)

            # Co-Occurrence
            if is_cooccurrence_dataset:
                parts = re.findall(r'cooccurrence_(.+?)_(.+?)_', result_file)[0]
                assert len(parts) == 2
                result['window_size'], result['words'] = parts
            # Concept Maps
            else:
                result['words'] = 'concepts'
        # DUMMY
        elif 'dummy' in result_file:
            result['type'] = 'dummy'
            result['words'] = 'dummy'
        # TEXT
        else:
            result['type'] = 'text'
            result['words'] = ['all'] * num_results

        if 'time_checkpoints' in result_data:
            timestamps = result_data['time_checkpoints']
            timestamps = sorted(timestamps.items(), key=lambda x: x[1])

            start = timestamps[0][1]
            end = timestamps[-1][1]

            result['timestamps'] = [timestamps] * num_results
            result['time'] = [end - start] * num_results

        result['filename'] = result_file
        result['dataset'] = dataset_name

        # Add meta data
        info = {}
        if 'results' in result_data:
            info = {'info__' + k: v for k, v in result_data.get('meta_data', result_data).items() if k != 'results'}
        result = dict(result, **{k: [v] * num_results for k, v in info.items()})

        data_.append(result)

    df_all = None
    for d in data_:
        result_df = pd.DataFrame(d)
        df_all = result_df if df_all is None else df_all.append(result_df)

    if df_all is None or not len(df_all):
        LOGGER.warning('Did not retrieve results! Aborting')
        return None

    # Remove cols
    df_all = df_all[[
        x for x in df_all.columns.tolist() if
        (not remove_split_cols or not re.match(r'^split\d', x)) and
        (not remove_fit_time_cols or not re.match(r'_time$', x)) and
        (not remove_rank_cols or not re.match(r'rank_', x))
    ]]

    # Change the column order
    prio_columns = ['dataset', 'type', 'combined']
    low_prio_columns = ['params', 'filename'] + [c for c in df_all.columns if c.startswith('std_') or c.startswith('mean_')]
    columns = df_all.columns.tolist()
    for c in prio_columns + low_prio_columns:
        columns.remove(c)

    return df_all.reset_index(drop=True)[prio_columns + columns + low_prio_columns]


def calculate_scores(Y_true, Y_pred, metrics=significance_test_utils.metrics):
    results = {}
    for metric_name, fn in metrics:
        results[metric_name] = fn(Y_true, Y_pred)
    return results


def get_result_folders(results_directory=constants.RESULTS_FOLDER):
    return sorted([x for x in glob('{}/201*'.format(results_directory)) if os.path.isdir(x) and len(glob('{}/*.npy'.format(x)))])


def get_result_filenames_from_folder(folder: str = None):
    if not folder:
        folder = get_result_folders()[-1]
    return glob('{}/*.npy'.format(folder))


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
        if filenames and p_filename not in filenames:
            continue
        with open(prediction_file, 'rb') as f:
            prediction = pickle.load(f)
        yield prediction_file, prediction


def cleanup_outdated_predictions(results_folder: str = None, dry_run=True, ignore_filter: str = 'dummy'):
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
        df_ = get_results(filter_out_experiment=x, **get_results_kwargs)
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


def _get_experiment_name_from_filename(x):
    filename = filename_utils.get_filename_only(x, with_extension=False)
    parts = filename.split('__')
    if 'result___experiment' not in x or len(parts) < 4: return ''
    return parts[1][1:].strip()


def remove_transformer_classes(d):
    def get_typename(val):
        return type(val).__name__

    if isinstance(d, dict):
        for key, val in d.items():
            if val is None:
                continue
            if isinstance(val, (np.ma.masked_array, list)):
                for idx, x in enumerate(val):
                    if isinstance(x, sklearn.base.BaseEstimator):
                        val[idx] = get_typename(x)
                    if isinstance(x, dict):
                        remove_transformer_classes(x)
            if isinstance(val, sklearn.base.BaseEstimator):
                d[key] = get_typename(val)
            if callable(val):
                d[key] = val.__name__
            if isinstance(val, dict):
                remove_transformer_classes(val)


replacements = [
    ('param_', ''),
    ('features__fast_wl_pipeline__feature_extraction__feature_extraction__', 'graph__'),
    ('features__fast_wl_pipeline__feature_extraction__', ''),
    ('graph_preprocessing', 'graph__preprocessing'),
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
    if 'combined' in filename or '__graph__' in filename or 'graph_structure_only' in filename or 'graph_relabel' in filename or '_graph_extra_' in filename:
        parts.append('wl')
    if 'node_weight' in filename:
        parts.append('wl_nodeweight')

    if not len(parts):
        raise Exception('Unknown kernel: {}'.format(filename))

    return '_'.join(parts)
