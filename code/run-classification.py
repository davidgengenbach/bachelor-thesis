#!/usr/bin/env python

import pandas as pd
import helper
import dataset_helper
import numpy as np
import sklearn
import graph_helper
import wl
import os
import pickle
import json
import tempfile
import gc
import traceback
import logging
from sklearn import dummy
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from transformers.fast_wl_graph_kernel_transformer import FastWLGraphKernelTransformer
from transformers.phi_picker_transformer import PhiPickerTransformer
from transformers.wl_graph_kernel_transformer import WLGraphKernelTransformer
from transformers.preprocessing_transformer import PreProcessingTransformer

from logger import LOGGER
from remove_coefs_from_results import remove_coefs_from_results 


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Run classification on the text and graph datasets')
    parser.add_argument('--n_jobs', type=int, default=2)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--verbose', type=int, default=11)
    parser.add_argument('--check_texts', action="store_true")
    parser.add_argument('--check_graphs', action="store_true")
    parser.add_argument('--remove_coefs', action="store_true")
    parser.add_argument('--max_iter', type=int, default = 1000)
    parser.add_argument('--tol', type=int, default = 1e-3)
    parser.add_argument('--n_splits', type=int, default = 3)
    parser.add_argument('--random_state', type=int, default = 42)
    parser.add_argument('--limit_dataset', nargs='+', type=str, default=['ng20', 'ling-spam', 'reuters-21578', 'webkb'], dest='limit_dataset')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    scoring = ['precision_macro', 'recall_macro', 'accuracy', 'f1_macro']
    refit = 'f1_macro'

    cv = sklearn.model_selection.StratifiedKFold(
        n_splits=args.n_splits,
        random_state=args.random_state,
        shuffle=True
    )

    clfs = [
        sklearn.dummy.DummyClassifier(strategy='most_frequent'),
        #sklearn.svm.SVC(max_iter = args.max_iter, tol=args.tol),
        #sklearn.linear_model.Perceptron(class_weight='balanced', max_iter=args.max_iter, tol=args.tol),
        #sklearn.linear_model.LogisticRegression(class_weight = 'balanced', max_iter=args.max_iter, tol=args.tol),
        sklearn.linear_model.PassiveAggressiveClassifier(class_weight='balanced', max_iter=args.max_iter, tol=args.tol)
    ]

    LOGGER.info('{:<10} - Starting'.format('Text'))
    if args.check_texts:
        for dataset_name in dataset_helper.get_all_available_dataset_names():
            LOGGER.info('{:<10} - {:<15}'.format('Text', dataset_name))
            if args.limit_dataset and dataset_name not in args.limit_dataset:
                continue

            result_file = 'data/results/text_{}.results.npy'.format(dataset_name)

            if not args.force and os.path.exists(result_file):
                continue

            gc.collect()

            LOGGER.info('{:<10} - {:<15} - Retrieving dataset'.format('Text', dataset_name))
            X, Y = dataset_helper.get_dataset(dataset_name, use_cached=True)

            p = Pipeline([
                ('preprocessing', None),
                ('count_vectorizer', sklearn.feature_extraction.text.CountVectorizer()),
                ('TfidfTransformer', sklearn.feature_extraction.text.TfidfTransformer()),
                ('scaler', sklearn.preprocessing.StandardScaler(with_mean = False)),
                ('clf', None)
            ])

            param_grid = dict(
                preprocessing= [None, PreProcessingTransformer(only_nouns=True)],
                count_vectorizer__stop_words=['english'],
                clf=clfs
            )

            gscv = GridSearchCV(estimator=p, param_grid=param_grid, cv=cv,
                                scoring=scoring, n_jobs=args.n_jobs, verbose=args.verbose, refit=refit)

            LOGGER.info('{:<10} - {:<15} - Starting to fit'.format('Text', dataset_name))

            gscv_result = gscv.fit(X, Y)

            if args.remove_coefs:
                remove_coefs_from_results(gscv_result.cv_results_)

            with open(result_file, 'wb') as f:
                pickle.dump(gscv_result.cv_results_, f)
            LOGGER.info('{:<10} - {:<15} - Finished'.format('Text', dataset_name))
        LOGGER.info('{:<10} - Finished'.format('Text'))

    if args.check_graphs:
        LOGGER.info('{:<10} - Starting'.format('Graph'))
        for cache_file in dataset_helper.get_all_cached_graph_phi_datasets():
            dataset = dataset_helper.get_dataset_name_from_graph_cachefile(cache_file)
            if args.limit_dataset and dataset not in args.limit_dataset:
                continue

            graph_dataset_cache_file = cache_file.split('/')[-1]

            LOGGER.info('{:<10} - {:<15}'.format('Graph', graph_dataset_cache_file))

            gc.collect()

            LOGGER.info('{:<10} - {:<15} - Retrieving dataset'.format('Graph', graph_dataset_cache_file))
            X_all, Y = dataset_helper.get_dataset_cached(cache_file, check_validity=False)

            for h in range(len(X_all)):
                LOGGER.info('{:<10} - {:<15} - Classifying for h={}'.format('Graph', graph_dataset_cache_file, h))
                result_file = 'data/results/{}.{}.results.npy'.format(graph_dataset_cache_file, h)
                if not args.force and os.path.exists(result_file):
                    LOGGER.warning('\tAlready calculated result: {}'.format(result_file))
                    continue

                X = X_all[h]

                p = Pipeline([
                    ('scaler', None),
                    ('clf', None)
                ])

                param_grid = dict(
                    scaler=[None, sklearn.preprocessing.Normalizer(norm="l1", copy=False)],
                    clf=clfs,
                )

                gscv = GridSearchCV(
                    estimator=p,
                    param_grid=param_grid,
                    cv=cv,
                    scoring=scoring,
                    refit=refit,
                    n_jobs=args.n_jobs,
                    verbose=args.verbose
                )

                try:
                    LOGGER.info('{:<10} - {:<15} - Classifying for h={}, fitting'.format('Graph', graph_dataset_cache_file, h))
                    gscv_result = gscv.fit(X, Y)


                    if args.remove_coefs:
                        remove_coefs_from_results(gscv_result.cv_results_)

                    with open(result_file, 'wb') as f:
                        pickle.dump(gscv_result.cv_results_, f)
                except Exception as e:
                    LOGGER.warning('{:<10} - {:<15} - Error h={}'.format('Graph', graph_dataset_cache_file, h))
                    LOGGER.exception(e)
                    continue
                LOGGER.info('{:<10} - {:<15} - Finished for h={}'.format('Graph', graph_dataset_cache_file, h))
    LOGGER.info('Finished!')

if __name__ == '__main__':
    main()
