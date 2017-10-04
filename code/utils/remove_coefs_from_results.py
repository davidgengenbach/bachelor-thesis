#!/usr/bin/env python
from glob import glob
import pickle
from pprint import pprint
import os
import numpy as np

def remove_coefs(clf):
    found = False
    for x in ['coef_', 'class_log_prior_', 'intercept_', 'feature_log_prob_', 'class_count_', 'feature_count_']:
        try:
            delattr(clf, x)
            found = True
        except:
            pass

    return found

def remove_coefs_from_results(results):
    found = False
    for attr in ['param_clf', 'param_classifier']:
        if attr not in results: continue
        param_clf = results[attr]
        if np.ma.isMaskedArray(param_clf):
            param_clf = np.ma.asarray(param_clf)

        if not isinstance(param_clf, list) and not isinstance(param_clf, (np.ndarray, np.generic)) and not np.ma.isMaskedArray(param_clf):
            param_clf = [param_clf]

        for clf in param_clf:
            found |= remove_coefs(clf)

    return found