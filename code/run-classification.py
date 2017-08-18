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
    parser.add_argument('--scoring', type=str, default="f1_macro")
    parser.add_argument('--verbose', type=int, default=11)
    parser.add_argument('--filter_dataset', type=str,
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

    for dataset_name in dataset_helper.get_all_available_dataset_names():
        # TODO
        break
        if args.filter_dataset and not args.filter_dataset in dataset_name:
            continue
        result_file = 'data/results/text_{}.results.npy'.format(dataset_name)

        if os.path.exists(result_file):
            continue
        gc.collect()

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
            clf=[sklearn.linear_model.PassiveAggressiveClassifier(max_iter=1000)],
            clf__class_weight=['balanced']
        )

        gscv = GridSearchCV(estimator=p, param_grid=param_grid, cv=cv,
                            scoring=args.scoring, n_jobs=args.n_jobs, verbose=11)
        gscv_result = gscv.fit(X, Y)

        with open(result_file, 'wb') as f:
            pickle.dump(gscv_result.cv_results_, f)
        logger.info('Best score:\t{:.5f}\nBest params:\t{}'.format(gscv_result.best_score_, gscv_result.best_params_))

    for cache_file in dataset_helper.get_all_cached_graph_phi_datasets():
        if args.filter_dataset and not args.filter_dataset in cache_file:
            continue

        graph_dataset_cache_file = cache_file.split('/')[-1]

        logger.info('{}\tDataset File: {}'.format('#' * 10, graph_dataset_cache_file))

        if not os.path.exists(cache_file):
            logger.warning('\tCould not find cachefile: "{}". Skipping.'.format(cache_file))
            continue
        gc.collect()

        X_all, Y = dataset_helper.get_dataset_cached(cache_file, check_validity=False)

        for h in range(len(X_all)):
            result_file = 'data/results/{}.{}.results.npy'.format(graph_dataset_cache_file, h)
            if os.path.exists(result_file):
                logger.warning('\tAlready calculated result: {}'.format(result_file))
                continue
            with open(result_file, 'w') as f:
                f.write('NOT DONE')

            X = [x.T for x in X_all[h]]

            p = Pipeline([
                #('scaler', None),
                ('clf', None)
            ])

            param_grid = dict(
                #scaler = [None, sklearn.preprocessing.Normalizer(norm="l1", copy = False)],
                clf=[sklearn.linear_model.PassiveAggressiveClassifier(max_iter=1000, class_weight='balanced')]
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
                gscv_result = gscv.fit(X, Y)

                if hasattr(param_grid['clf'], 'coef_'):
                    del param_grid['clf'].coef_

                with open(result_file, 'wb') as f:
                    pickle.dump(gscv_result.cv_results_, f)
            except Exception as e:
                logger.exception(e)
                continue

if __name__ == '__main__':
    main()
