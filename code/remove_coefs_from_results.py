#!/usr/bin/env python
from glob import glob
import pickle
from pprint import pprint
import os

def remove_coefs(clf):
    if hasattr(clf, 'coef_'):
        del clf.coef_
        return True
    return False

for result_file in glob('data/results/*.npy'):
    if 'model_removed' in result_file: continue
    if 'text_' not in result_file: continue
    print(result_file)
    try:
        with open(result_file, 'rb') as f:
            results = pickle.load(f)
        param_clf = results['param_clf']
        if not isinstance(param_clf, list):
            param_clf = [param_clf]
        found = False
        for clf in param_clf:
            found |= remove_coefs(clf)
            
        if found:
            with open(result_file, 'wb') as f:
                pickle.dump(results, f)
    except Exception as e:
        print('\tError: {}'.format(e))
