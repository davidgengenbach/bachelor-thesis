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

for dataset_name in dataset_helper.get_all_available_dataset_names():
    result_file = 'data/results/text_{}.results.npy'.format(dataset_name)

    if os.path.exists(result_file): continue
    gc.collect()
    
    X, Y = dataset_helper.get_dataset(dataset_name, use_cached= True)
    
    p = Pipeline([
        ('preprocessing', None),
        ('count_vectorizer', sklearn.feature_extraction.text.CountVectorizer()),
        ('TfidfTransformer', sklearn.feature_extraction.text.TfidfTransformer()),
        ('clf', None)
    ])
    
    param_grid = dict(
        preprocessing = [None, PreProcessingTransformer(only_nouns = True)],
        count_vectorizer__stop_words = ['english'],
        clf = [sklearn.linear_model.PassiveAggressiveClassifier(max_iter = 1000)],
        clf__class_weight = ['balanced']
    )

    cv = sklearn.model_selection.StratifiedKFold(n_splits = 3, random_state= 42, shuffle= True)
    gscv = GridSearchCV(estimator = p, param_grid=param_grid, cv=cv, scoring = 'f1_macro', n_jobs=1, verbose = 11)
    gscv_result = gscv.fit(X, Y)
    
    with open(result_file, 'wb') as f:
        pickle.dump(gscv_result.cv_results_, f)
    logger.info('Best score:\t{:.5f}\nBest params:\t{}'.format(gscv_result.best_score_, gscv_result.best_params_))

for cache_file in dataset_helper.get_all_cached_graph_datasets():
    graph_dataset_cache_file = cache_file.split('/')[-1]
    
    result_file = 'data/results/{}.results.npy'.format(graph_dataset_cache_file)
    logger.info('{}\tDataset File: {}'.format('#' * 10, graph_dataset_cache_file))

    if os.path.exists(result_file):
        logger.warning('\tAlready calculated result: {}'.format(result_file))
        continue

    if not os.path.exists(cache_file):
        logger.warning('\tCould not find cachefile: "{}". Skipping.'.format(cache_file))
        continue
    gc.collect()
    
    X, Y = dataset_helper.get_dataset_cached(cache_file)
    X, Y = np.array(X), np.array(Y)
    
    p = Pipeline([
        ('wl_transformer', FastWLGraphKernelTransformer()),
        ('phi_picker', PhiPickerTransformer()),
        ('scaler', None),
        ('clf', None)
    ])

    param_grid = dict(
        wl_transformer__h=[2],
        phi_picker__return_iteration=[0, 1, 2],
        scaler = [None, sklearn.preprocessing.Normalizer(norm="l1", copy = False)],
        clf = [sklearn.linear_model.PassiveAggressiveClassifier(max_iter = 1000)],
        clf__max_iter=[1000],
        clf__tol = [1e-3],
        clf__class_weight=['balanced']
    )

    cv = sklearn.model_selection.StratifiedKFold(n_splits = 3, random_state= 42, shuffle= True)
    gscv = GridSearchCV(
        estimator = p,
        param_grid=param_grid,
        cv=cv,
        scoring = 'f1_macro',
        n_jobs=1,
        verbose = 11
    )
    if 1 == 1:
        try:
            gscv_result = gscv.fit(X, Y)
        except Exception as e:
            logger.exception(e)
            continue
    else:
        gscv_result = gscv.fit(X, Y)
    with open(result_file, 'wb') as f:
        pickle.dump(gscv_result.cv_results_, f)