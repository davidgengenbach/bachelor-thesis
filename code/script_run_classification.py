#!/usr/bin/env python

import gc
import os
import sys
import pickle
from glob import glob
import tempfile
import shutil
import collections
from time import time
import typing

from joblib import Parallel, delayed
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn import base
from sklearn import dummy
from sklearn import naive_bayes
from sklearn.base import clone
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from transformers import text_pipeline
from transformers import fast_wl_pipeline
from transformers.phi_picker_transformer import PhiPickerTransformer
from transformers.fast_wl_graph_kernel_transformer import FastWLGraphKernelTransformer
from transformers.nx_graph_to_tuple_transformer import NxGraphToTupleTransformer
from transformers.tuple_selector import TupleSelector

from utils.logger import LOGGER
from utils.remove_coefs_from_results import remove_coefs_from_results
from utils import dataset_helper, filename_utils, time_utils, helper

Task = collections.namedtuple('Task', ['type', 'name', 'process_fn', 'process_fn_args'])

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Run classification on the text and graph datasets')
    parser.add_argument('--n_jobs', type=int, default=2)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--verbose', type=int, default=11)
    parser.add_argument('--create_predictions', action="store_true")
    parser.add_argument('--keep_coefs', action="store_true")
    parser.add_argument('--dry_run', action="store_true")
    parser.add_argument('--wl_iterations', nargs='+', type=int, default=[4])
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--tol', type=int, default=6e-4)
    parser.add_argument('--n_splits', type=int, default=3)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--task_name_filter', type=str, default=None)
    parser.add_argument('--task_type_include_filter', type=str, nargs='+', default=None)
    parser.add_argument('--task_type_exclude_filter', type=str, nargs='+', default=None)
    parser.add_argument('--wl_round_to_decimal', nargs='+', type=int, default=[-1, 0, 1, 10])
    parser.add_argument('--limit_dataset', nargs='+', type=str, default=['ng20', 'ling-spam', 'reuters-21578', 'webkb', 'webkb-ana', 'ng20-ana'])
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    helper.print_script_args_and_info(args)

    scoring = ['precision_macro', 'recall_macro', 'accuracy', 'f1_macro']
    refit = 'f1_macro'

    tasks = []


    RESULTS_FOLDER = 'data/results'
    PREDICTIONS_FOLDER = '{}/predictions'.format(RESULTS_FOLDER)

    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(PREDICTIONS_FOLDER, exist_ok=True)

    cv = sklearn.model_selection.StratifiedKFold(
        n_splits=args.n_splits,
        random_state=args.random_state,
        shuffle=True
    )

    clfs = [
        #sklearn.dummy.DummyClassifier(strategy='most_frequent'),
        sklearn.naive_bayes.MultinomialNB(),
        sklearn.svm.LinearSVC(class_weight='balanced', max_iter=args.max_iter, tol=args.tol),
        sklearn.linear_model.PassiveAggressiveClassifier(class_weight='balanced', max_iter=args.max_iter, tol=args.tol)
        #sklearn.naive_bayes.GaussianNB(),
        #sklearn.svm.SVC(max_iter = args.max_iter, tol=args.tol),
        #sklearn.linear_model.Perceptron(class_weight='balanced', max_iter=args.max_iter, tol=args.tol),
        #sklearn.linear_model.LogisticRegression(class_weight = 'balanced', max_iter=args.max_iter, tol=args.tol),
    ]

    def cross_validate(task: Task, X, Y, estimator, param_grid: dict):
        result_filename_tmpl = filename_utils.get_result_filename_for_task(task)

        result_file = '{}/{}'.format(RESULTS_FOLDER, result_filename_tmpl)
        predictions_file = '{}/{}'.format(PREDICTIONS_FOLDER, result_filename_tmpl)


        X_train, Y_train, X_test, Y_test = X, Y, [], []

        # Hold out validation set (15%)
        if args.create_predictions:
            try:
                X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, stratify=Y, test_size=0.15)
            except Exception as e:
                LOGGER.warning('Could not split dataset for predictions')
                LOGGER.exception(e)

        gscv = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=args.n_jobs, verbose=args.verbose, refit=refit)

        gscv_result = gscv.fit(X_train, Y_train)

        if args.create_predictions:
            if not len(X_test) or len(Y_test):
                LOGGER.warning('Validation set for prediction has no length: len(X_test)={}'.format(len(X_test)))
            else:
                try:
                    # Retrain the best classifier and get prediction on validation set
                    best_classifer = sklearn.base.clone(gscv_result.best_estimator_)
                    best_classifer.fit(X_train, Y_train)
                    Y_test_pred = best_classifer.predict(X_test)

                    with open(predictions_file, 'wb') as f:
                        pickle.dump({
                            'Y_real': Y_test,
                            'Y_pred': Y_test_pred
                        }, f)
                except Exception as e:
                    LOGGER.warning('Error while trying to retrain best classifier')
                    LOGGER.exception(e)

        if not args.keep_coefs:
            remove_coefs_from_results(gscv_result.cv_results_)

        with open(result_file, 'wb') as f:
            pickle.dump(gscv_result.cv_results_, f)

    # Text classification
    for dataset_name in dataset_helper.get_all_available_dataset_names():

        def process(task, dataset_name):
            X, Y = dataset_helper.get_dataset(dataset_name)
            cross_validate(task, X, Y, text_pipeline.get_pipeline(), dict(text_pipeline.get_param_grid(), **dict(classifier=clfs)))

        tasks.append(Task(type='text', name=dataset_name, process_fn=process, process_fn_args=[dataset_name]))

    graph_fast_wl_classification_pipeline = sklearn.pipeline.Pipeline([
        ('feature_extraction', fast_wl_pipeline.get_pipeline()),
        ('classifier', None)
    ])

    graph_fast_wl_grid_params = {
        'fast_wl__h': args.wl_iterations,
        'fast_wl__phi_dim': [None],
        'fast_wl__round_to_decimals': args.wl_round_to_decimal,
        'phi_picker__return_iteration': [0, 1, 2, 3]
    }

    # fast_wl graph and combined with text classification
    for graph_cache_file in dataset_helper.get_all_cached_graph_datasets():
        filename = filename_utils.get_filename_only(graph_cache_file)
        def process(task, graph_cache_file):
            X, Y = dataset_helper.get_dataset_cached(graph_cache_file)

            empty_graphs = [1 for g in X if nx.number_of_nodes(g) == 0 or nx.number_of_edges(g) == 0]
            num_vertices = sum([nx.number_of_nodes(g) for g in X]) + len(empty_graphs)

            fast_wl_pipeline.convert_graphs_to_tuples(X)

            features_params = dict({'feature_extraction__' + k: val for k, val in
                                    graph_fast_wl_grid_params.items()}, **dict(
                feature_extraction__fast_wl__phi_dim=[num_vertices]
            ))

            grid_params_scratch = dict(dict(classifier = clfs), **features_params)

            cross_validate(task, X, Y, graph_fast_wl_classification_pipeline, grid_params_scratch)

        def process_combined(task, graph_cache_file):
            X, Y = dataset_helper.get_dataset_cached(graph_cache_file)
            X_text, Y_text = dataset_helper.get_dataset(dataset_name)

            empty_graphs = [1 for g in X if nx.number_of_nodes(g) == 0 or nx.number_of_edges(g) == 0]
            num_vertices = sum([nx.number_of_nodes(g) for g in X]) + len(empty_graphs)

            fast_wl_pipeline.convert_graphs_to_tuples(X)

            grid_params_combined = dict({
                'classifier': clfs
            }, **dict({'features__fast_wl_pipeline__feature_extraction__' + k: val for k, val in graph_fast_wl_grid_params.items()}, **dict(
                features__fast_wl_pipeline__feature_extraction__fast_wl__phi_dim = [num_vertices]
            )))

            combined_features = sklearn.pipeline.FeatureUnion([
                ('tfidf', sklearn.pipeline.Pipeline([
                    ('selector', TupleSelector(tuple_index=0)),
                    ('tfidf', text_pipeline.get_pipeline()),
                ])),
                ('fast_wl_pipeline', sklearn.pipeline.Pipeline([
                    ('selector', TupleSelector(tuple_index=1, v_stack=False)),
                    ('feature_extraction', fast_wl_pipeline.get_pipeline())
                ]))
            ])

            pipeline = sklearn.pipeline.Pipeline([
                ('features', combined_features),
                ('classifier', None)
            ])

            X_combined = list(zip(X_text, X))
            cross_validate(X_combined, Y, pipeline, grid_params_combined, result_file_combined, predictions_file_combined)

        tasks.append(Task(type='graph_fast_wl', name = filename, process_fn = process, process_fn_args=[graph_cache_file]))
        tasks.append(Task(type='graph_combined-fast_wl', name= filename, process_fn=process_combined, process_fn_args=[graph_cache_file]))

    # Gram classification
    for gram_cache_file in glob('data/CACHE/*gram*.npy'):
        def process(task, gram_cache_file):
            K, Y = dataset_helper.get_dataset_cached(gram_cache_file, check_validity=False)
            estimator = Pipeline([('clf', sklearn.svm.SVC(kernel='precomputed', class_weight='balanced', max_iter=args.max_iter, tol=args.tol))])
            cross_validate(K, Y, estimator, {})

        filename = filename_utils.get_filename_only(gram_cache_file)
        tasks.append(Task(type='graph-gram', name=filename, process_fn=process, process_fn_args=[graph_cache_file]))

    start_tasks(args, tasks)


def start_tasks(args, tasks: typing.List[Task]):

    def should_process_task(task: Task):
        # Dataset filter
        is_filtered_by_dataset = args.limit_dataset and filename_utils.get_dataset_from_filename(task.name) not in args.limit_dataset

        # Task type filters
        is_filtered_by_include_filter = (args.task_type_include_filter and task.type not in args.task_type_include_filter)
        is_filtered_by_exclude_filter = (args.task_type_exclude_filter and task.type in args.task_type_exclude_filter)

        is_filtered_by_name_filter = (args.task_name_filter and args.task_name_filter not in task.name)

        # Do not process tasks that have already been calculated (unless args.force == True)
        created_files = [filename_utils.get_result_filename_for_task(task)]
        is_filtered_by_file_exists = (args.force and np.any([os.path.exists(file) for file in created_files]))

        should_process = not np.any([
            is_filtered_by_dataset,
            is_filtered_by_include_filter,
            is_filtered_by_name_filter,
            is_filtered_by_file_exists,
            is_filtered_by_exclude_filter
        ])

        return should_process

    def print_tasks(tasks):
        for task in sorted(tasks, key = lambda x: x.type):
            print('\t{t.type:26} {dataset:18} {t.name}'.format(t=task, dataset = filename_utils.get_dataset_from_filename(task.name)))
        print('\n')

    if args.dry_run:
        print('All tasks:')
        print_tasks(tasks)

    task_type_counter_unfiltered = collections.Counter([t.type for t in tasks])

    # Filter out tasks
    tasks = sorted([task for task in tasks if should_process_task(task)], key = lambda x: x.type)

    print('Filtered tasks:')
    print_tasks(tasks)

    print('# tasks per type (filtered/unfiltered)')
    task_type_counter_filtered = collections.Counter([t.type for t in tasks])
    for task_type, count in task_type_counter_unfiltered.items():
        print('\t{:25} {:2}/{:2}'.format(task_type, task_type_counter_filtered[task_type], count))
    print('\n')

    if args.dry_run:
        print('Only doing a dry-run. Exiting.')
        return

    num_tasks = len(tasks)
    for task_idx, t in enumerate(tasks):
        def print_task(task, msg = ''):
            LOGGER.info('Task {idx:>2}/{num_tasks}: {t.type:30} - {t.name:40} - {msg}'.format(idx = task_idx + 1, num_tasks = num_tasks, t = task, msg = msg))

        start_time = time()
        print_task(t, 'Started')
        try:
            t.process_fn(t, *t.process_fn_args)
        except Exception as e:
            print_task(t, 'Error: {}'.format(e))
            LOGGER.exception(e)
        elapsed_seconds = time() - start_time
        print_task(t, 'Finished (time={})'.format(time_utils.seconds_to_human_readable(elapsed_seconds)))

    LOGGER.info('Finished!')


if __name__ == '__main__':
    main()
