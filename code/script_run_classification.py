#!/usr/bin/env python

import gc
import os
import sys
import pickle
from glob import glob

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


from transformers.phi_picker_transformer import PhiPickerTransformer
from transformers.fast_wl_graph_kernel_transformer import FastWLGraphKernelTransformer
from transformers.nx_graph_to_tuple_transformer import NxGraphToTupleTransformer
from transformers.tuple_selector import TupleSelector
from utils import dataset_helper
from utils.logger import LOGGER
from utils.remove_coefs_from_results import remove_coefs_from_results
from utils import filter_utils
from utils import filename_utils

import networkx as nx


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Run classification on the text and graph datasets')
    parser.add_argument('--n_jobs', type=int, default=2)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--verbose', type=int, default=11)
    parser.add_argument('--check_texts', action="store_true")
    parser.add_argument('--check_graphs', action="store_true")
    parser.add_argument('--check_combined', action="store_true")
    parser.add_argument('--check_gram', action="store_true")
    parser.add_argument('--check_graphs_scratch', action="store_true")
    parser.add_argument('--create_predictions', action="store_true")
    parser.add_argument('--remove_coefs', action="store_true")
    parser.add_argument('--wl_iterations', type=int, default=4)
    parser.add_argument('--max_iter', type=int, default=1000)
    parser.add_argument('--tol', type=int, default=1e-3)
    parser.add_argument('--n_splits', type=int, default=3)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--include_graphs', type=str, default=None)
    parser.add_argument('--exclude_graphs', type=str, default=None)
    parser.add_argument('--limit_dataset', nargs='+', type=str, default=[
                        'ng20', 'ling-spam', 'reuters-21578', 'webkb', 'webkb-ana', 'ng20-ana'])
    args = parser.parse_args()
    return args


def file_should_be_processed(file, dataset, include_filter, exclude_filter, limit_dataset):
    """Returns true, if file is included AND not excluded AND in the limited datasets.
    
    Args:
        file (str): the file to be processed
        dataset (str): the dataset of the file
        include_filter (str): string that has to be in `file` (can be None)
        exclude_filter (str): string that must not be in `file` (can be None)
        limit_dataset (list(str)): the datasets that have been limited (= are allowed)
    
    Returns:
        bool: Whether the file should be processed
    """
    is_in_limited_datasets = (not limit_dataset or dataset in limit_dataset)
    is_included = (not include_filter or include_filter in file)
    is_excluded = (exclude_filter and exclude_filter in file)
    return is_in_limited_datasets and is_included and not is_excluded


def main():
    args = get_args()

    scoring = ['precision_macro', 'recall_macro', 'accuracy', 'f1_macro']
    refit = 'f1_macro'

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
        #sklearn.linear_model.PassiveAggressiveClassifier(class_weight='balanced', max_iter=args.max_iter, tol=args.tol)
        #sklearn.naive_bayes.GaussianNB(),
        #sklearn.svm.SVC(max_iter = args.max_iter, tol=args.tol),
        #sklearn.linear_model.Perceptron(class_weight='balanced', max_iter=args.max_iter, tol=args.tol),
        #sklearn.linear_model.LogisticRegression(class_weight = 'balanced', max_iter=args.max_iter, tol=args.tol),
    ]

    def cross_validate(X, Y, estimator, param_grid, result_file, predictions_file, create_predictions):
        gc.collect()

        X_train, Y_train, X_test, Y_test = X, Y, [], []

        # Hold out validation set (15%)
        if create_predictions:
            try:
                X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, Y, stratify=Y, test_size=0.15)
            except Exception as e:
                LOGGER.warning('Could not split dataset for predictions')
                LOGGER.exception(e)

        gscv = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=cv, scoring=scoring, n_jobs=args.n_jobs, verbose=args.verbose, refit=refit)

        gscv_result = gscv.fit(X_train, Y_train)

        if create_predictions:
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

        if args.remove_coefs:
            remove_coefs_from_results(gscv_result.cv_results_)

        with open(result_file, 'wb') as f:
            pickle.dump(gscv_result.cv_results_, f)

    if args.check_texts:
        LOGGER.info('{:<10} - Starting'.format('Text'))
        for dataset_name in dataset_helper.get_all_available_dataset_names():
            if not filter_utils.file_should_be_processed(dataset_name, None, None, args.limit_dataset):
                continue

            result_file = '{}/text_{}.results.npy'.format(RESULTS_FOLDER, dataset_name)
            predictions_file = '{}/text_{}.npy'.format(PREDICTIONS_FOLDER, dataset_name)

            if not args.force and os.path.exists(result_file):
                continue

            LOGGER.info('{:<10} - {:<15} - Starting'.format('Text', dataset_name))
            X, Y = dataset_helper.get_dataset(dataset_name, use_cached=True)

            estimator = Pipeline([
                ('preprocessing', None),
                ('TfidfTransformer', sklearn.feature_extraction.text.TfidfVectorizer(
                    stop_words='english')),
                ('scaler', None),
                ('clf', None)
            ])

            param_grid = dict(
                #preprocessing= [None, PreProcessingTransformer(only_nouns=True, return_lemma = True)],
                #scaler = [None, sklearn.preprocessing.StandardScaler(with_mean = False)]
                clf=clfs
            )

            cross_validate(X, Y, estimator, param_grid, result_file, predictions_file, args.create_predictions)

            LOGGER.info('{:<10} - {:<15} - Finished'.format('Text', dataset_name))


    if args.check_graphs_scratch:
        for graph_cache_file in dataset_helper.get_all_cached_graph_datasets():
            dataset = filename_utils.get_dataset_from_filename(graph_cache_file)
            if not filter_utils.file_should_be_processed(graph_cache_file, args.include_graphs, args.exclude_graphs,  args.limit_dataset):
                continue
            filename = graph_cache_file.split('/')[-1]

            result_file = '{}/{}.scratch.results.npy'.format(RESULTS_FOLDER, filename)
            predictions_file = '{}/{}.scratch.npy'.format(PREDICTIONS_FOLDER, filename)

            X, Y = dataset_helper.get_dataset_cached(graph_cache_file)

            #X, Y = X[:20], Y[:20]
            empty_graphs = [1 for g in X if nx.number_of_nodes(g) == 0 or nx.number_of_edges(g) == 0]
            num_vertices = sum([nx.number_of_nodes(g) for g in X]) + len(empty_graphs)

            estimator = sklearn.pipeline.Pipeline([
                ('tuple_transformer', NxGraphToTupleTransformer()),
                ('fast_wl', FastWLGraphKernelTransformer(h=args.wl_iterations, should_cast=False, remove_missing_labels=True, phi_dim = num_vertices)),
                ('phi_picker', PhiPickerTransformer(return_iteration='stacked')),
                # ('gram_matrix', GramMatrixTransformer()),
                ('clf', None)
            ])

            param_grid = {
                'clf': clfs
            }

            try:
                LOGGER.info('{:<10} - {:<15} - Classifying'.format('Graph scratch', filename))
                cross_validate(X, Y, estimator, param_grid, result_file, predictions_file, args.create_predictions)
            except Exception as e:
                LOGGER.warning('{:<10} - {:<15} - Error'.format('Graph', filename))
                LOGGER.exception(e)




    if args.check_graphs:
        LOGGER.info('{:<10} - Starting'.format('Graph'))
        for cache_file in dataset_helper.get_all_cached_graph_phi_datasets():
            dataset = dataset_helper.get_dataset_name_from_graph_cachefile(cache_file)
            if not filter_utils.file_should_be_processed(cache_file, args.include_graphs, args.exclude_graphs,  args.limit_dataset):
                continue

            graph_dataset_cache_file = cache_file.split('/')[-1]

            LOGGER.info('{:<10} - {:<15}'.format('Graph', graph_dataset_cache_file))

            X_all, Y = dataset_helper.get_dataset_cached(cache_file, check_validity=False)

            # Ignore 0th iteration of WL when combining all WL iterations
            X_all.append(scipy.sparse.hstack(X_all[1:]))

            for h, X in enumerate(X_all):
                if h == len(X_all) - 1:
                    h = 'stacked'
                result_file = '{}/{}.{}.results.npy'.format(RESULTS_FOLDER, graph_dataset_cache_file, h)
                predictions_file = '{}/{}.{}.npy'.format(PREDICTIONS_FOLDER, graph_dataset_cache_file, h)

                if not args.force and os.path.exists(result_file):
                    continue

                LOGGER.info('{:<10} - {:<15} - Classifying for h={}'.format('Graph', graph_dataset_cache_file, h))

                estimator = Pipeline([
                    ('scaler', None),
                    ('clf', None)
                ])

                param_grid = dict(
                    scaler=[None, sklearn.preprocessing.Normalizer(norm="l1")],
                    clf=clfs,
                )

                try:

                    cross_validate(X, Y, estimator, param_grid, result_file, predictions_file, args.create_predictions)
                except Exception as e:
                    LOGGER.warning('{:<10} - {:<15} - Error h={}'.format('Graph', graph_dataset_cache_file, h))
                    LOGGER.exception(e)

    if args.check_gram:
        LOGGER.info('{:<10} - Starting'.format('Graph gram'))
        for gram_cache_file in glob('data/CACHE/*gram*.npy'):
            # TODO: add filter
            gram_cache_filename = gram_cache_file.split('/')[-1]
            result_file = '{}/{}.results.npy'.format(RESULTS_FOLDER, gram_cache_filename)
            predictions_file = '{}/{}.npy'.format(PREDICTIONS_FOLDER, gram_cache_filename)

            if not filter_utils.file_should_be_processed(gram_cache_filename, args.include_graphs, args.exclude_graphs, args.limit_dataset):
                continue

            # LOGGER.info('{:<10} - {:<15} - Classifying spgk, fitting'.format('Graph', gram_cache_filename))

            with open(gram_cache_file, 'rb') as f:
                K, Y = pickle.load(f)

            estimator = Pipeline([('clf', None)])

            param_grid = dict(
                clf=[sklearn.svm.SVC(kernel='precomputed', class_weight='balanced', max_iter=args.max_iter, tol=args.tol)]
            )

            try:
                cross_validate(K, Y, estimator, param_grid, result_file, predictions_file, args.create_predictions)
            except Exception as e:
                LOGGER.warning(
                    '{:<10} - {:<15} - Error spgk'.format('Graph', gram_cache_filename))
                LOGGER.exception(e)

    if args.check_combined:
        LOGGER.info('{:<10} - Starting'.format('Graph combined'))
        for dataset_name in dataset_helper.get_all_available_dataset_names():
            X_text, Y_text = dataset_helper.get_dataset(dataset_name)

            for graph_dataset_cache_file in dataset_helper.get_all_cached_graph_phi_datasets(dataset_name):

                if not filter_utils.file_should_be_processed(graph_dataset_cache_file, args.include_graphs, args.exclude_graphs, dataset_name, args.limit_dataset):
                    continue

                graph_dataset_cache_filename = graph_dataset_cache_file.split('/')[-1]
                X_phi, Y_phi = dataset_helper.get_dataset_cached(graph_dataset_cache_file, check_validity=False)
                for h, phi in enumerate(X_phi):
                    result_file = '{}/{}.combined.{}.results.npy'.format(RESULTS_FOLDER, graph_dataset_cache_filename, h)
                    predictions_file = '{}/combined.{}.{}.results.npy'.format(PREDICTIONS_FOLDER, graph_dataset_cache_filename, h)

                    if not args.force and os.path.exists(result_file):
                        continue

                    LOGGER.info('{:<10} - {:<15} ({}) h={}'.format('Graph combined', dataset_name, graph_dataset_cache_filename, h))

                    combined_features = sklearn.pipeline.FeatureUnion([
                        ('tfidf', sklearn.pipeline.Pipeline([
                            ('selector', TupleSelector(tuple_index=0)),
                            ('tfidf', sklearn.feature_extraction.text.TfidfVectorizer(stop_words='english')),
                        ])),
                        ('phi', sklearn.pipeline.Pipeline([
                            ('selector', TupleSelector(tuple_index=1, v_stack=True))
                        ]))
                    ])

                    param_grid = dict(clf=clfs)

                    estimator = sklearn.pipeline.Pipeline([
                        ("features", combined_features),
                        ("clf", None)
                    ])

                    if len(X_text) != phi.shape[0]:
                        LOGGER.warning('{:<10} - {:<15} - Error h={}, wrong dimensions, phi.shape[0]={}, len(X_text)={}'.format('Combined', graph_dataset_cache_filename, h, phi.shape[0], len(X_text)))
                        continue

                    try:
                        X_combined = list(zip(X_text, phi))
                        cross_validate(X_combined, Y_text, estimator, param_grid, result_file, predictions_file, args.create_predictions)
                    except Exception as e:
                        LOGGER.warning('{:<10} - {:<15} - Error h={}'.format('Combined', graph_dataset_cache_filename, h))
                        LOGGER.exception(e)
    LOGGER.info('Finished!')

if __name__ == '__main__':
    main()
