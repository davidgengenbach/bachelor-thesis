#!/usr/bin/env python
from glob import glob
import pickle
from pprint import pprint
import os
import numpy as np

def remove_coefs(clf):
    found = False
    for x in ['coef_', 'class_log_prior_', 'intercept_', 'feature_log_prob_', 'class_count_', 'feature_count_', 'vocabulary_', 'idf_', 'stop_words_', '_tfidf']:
        try:
            setattr(clf, x, None)
            found = True
        except:
            pass

    return found

def remove_coefs_from_results(results):
    found = False
    for attr, val in results.items():
        if np.ma.isMaskedArray(val):
            val = np.ma.asarray(val)

        if not isinstance(val, list) and not isinstance(val, (np.ndarray, np.generic)) and not np.ma.isMaskedArray(val):
            val = [val]

        for clf in val:
            found |= remove_coefs(clf)

    return found
