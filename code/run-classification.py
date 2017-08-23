#!/usr/bin/env python

import pandas as pd
import helper
import dataset_helper
import classifier_baseline
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
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron
from sklearn.model_selection import GridSearchCV
from transformers.fast_wl_graph_kernel_transformer import FastWLGraphKernelTransformer
from transformers.phi_picker_transformer import PhiPickerTransformer
from transformers.wl_graph_kernel_transformer import WLGraphKernelTransformer
from transformers.preprocessing_transformer import PreProcessingTransformer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Run classification on the text and graph datasets')
    parser.add_argument('--n_jobs', type=int, default=2)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--scoring', type=str, default="f1_macro")
    parser.add_argument('--verbose', type=int, default=11)
    parser.add_argument('--limit_dataset', type=str,
                        help="Only calculate datasets with this argument in the dataset name", default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    cv = sklearn.model_selection.StratifiedKFold(
        n_splits=3,
        random_state=42,
        shuffle=True
    )

    clfs = [
        sklearn.linear_model.Perceptron(max_iter=1000, tol=1e-4),
        sklearn.linear_model.LogisticRegression(max_iter=1000, tol=1e-4),
        sklearn.linear_model.SGDClassifier(max_iter=1000, tol=1e-4)
    ]

    print('{:<10} - Starting'.format('Text'))
    for dataset_name in dataset_helper.get_all_available_dataset_names():
        print('{:<10} - {:<15}'.format('Text', dataset_name))

        if args.limit_dataset and not args.limit_dataset in dataset_name:
            continue

        result_file = 'data/results/text_{}.results.npy'.format(dataset_name)
        if not args.force and os.path.exists(result_file):
            continue

        gc.collect()

        print('{:<10} - {:<15} - Retrieving dataset'.format('Text', dataset_name))
        X, Y = dataset_helper.get_dataset(dataset_name, use_cached=True)

        p = Pipeline([
            ('preprocessing', None),
            ('count_vectorizer', sklearn.feature_extraction.text.CountVectorizer()),
            ('TfidfTransformer', sklearn.feature_extraction.text.TfidfTransformer()),
            ('clf', None)
        ])

        param_grid = dict(
            preprocessing=[None, PreProcessingTransformer(only_nouns=True)],
            count_vectorizer__stop_words=['english'],
            clf=clfs,
            clf__class_weight=['balanced']
        )

        gscv = GridSearchCV(estimator=p, param_grid=param_grid, cv=cv,
                            scoring=args.scoring, n_jobs=args.n_jobs, verbose=args.verbose)

        print('{:<10} - {:<15} - Starting to fit'.format('Text', dataset_name))

        gscv_result = gscv.fit(X, Y)

        with open(result_file, 'wb') as f:
            pickle.dump(gscv_result.cv_results_, f)
        print('{:<10} - {:<15} - Finished'.format('Text', dataset_name))

    print('{:<10} - Finished'.format('Text'))
    print('{:<10} - Starting'.format('Graph'))
    for cache_file in dataset_helper.get_all_cached_graph_phi_datasets():
        if args.limit_dataset and not args.limit_dataset in cache_file:
            continue

        graph_dataset_cache_file = cache_file.split('/')[-1]

        print('{:<10} - {:<15}'.format('Graph', graph_dataset_cache_file))

        if not args.force and not os.path.exists(cache_file):
            print('{:<10} - {:<15} - Could not find cachefile: {}'.format('Graph', graph_dataset_cache_file, cache_file))
            continue
        gc.collect()

        print('{:<10} - {:<15} - Retrieving dataset'.format('Graph', graph_dataset_cache_file))
        X_all, Y = dataset_helper.get_dataset_cached(cache_file, check_validity=False)

        for h in range(len(X_all)):
            print('{:<10} - {:<15} - Classifying for h={}'.format('Graph', graph_dataset_cache_file, h))
            result_file = 'data/results/{}.{}.results.npy'.format(graph_dataset_cache_file, h)
            if os.path.exists(result_file):
                logger.warning('\tAlready calculated result: {}'.format(result_file))
                continue
            with open(result_file, 'w') as f:
                f.write('NOT DONE')

            X = X_all[h].T

            p = Pipeline([
                ('scaler', None),
                ('clf', None)
            ])

            param_grid = dict(
                scaler=[None, sklearn.preprocessing.Normalizer(norm="l1", copy=False)],
                clf=clfs,
                clf__class_weight = ['balanced']
            )

            gscv = GridSearchCV(
                estimator=p,
                param_grid=param_grid,
                cv=cv,
                scoring=args.scoring,
                n_jobs=args.n_jobs,
                verbose=args.verbose
            )

            try:
                print('{:<10} - {:<15} - Classifying for h={}, fitting'.format('Graph', graph_dataset_cache_file, h))
                gscv_result = gscv.fit(X, Y)

                if hasattr(param_grid['clf'], 'coef_'):
                    del param_grid['clf'].coef_

                with open(result_file, 'wb') as f:
                    pickle.dump(gscv_result.cv_results_, f)
            except Exception as e:
                logger.exception(e)
                continue
            print('{:<10} - {:<15} - Finished for h={}'.format('Graph', graph_dataset_cache_file, h))
    print('Finished!')

if __name__ == '__main__':
    main()
