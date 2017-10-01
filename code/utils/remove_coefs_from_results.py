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
    param_clf = results['param_clf']
    if np.ma.isMaskedArray(param_clf):
        param_clf = np.ma.asarray(param_clf)
    
    if not isinstance(param_clf, list) and not isinstance(param_clf, (np.ndarray, np.generic)) and not np.ma.isMaskedArray(param_clf):
        param_clf = [param_clf]

    found = False

    for clf in param_clf:
        found |= remove_coefs(clf)

    return found