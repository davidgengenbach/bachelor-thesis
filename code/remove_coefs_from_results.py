#!/usr/bin/env python
from glob import glob
import pickle
from pprint import pprint
import os
import numpy as np

def remove_coefs(clf):
    if hasattr(clf, 'coef_'):
        del clf.coef_
        return True
    return False

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

def main():
    for result_file in glob('data/results/*.npy'):
        if 'model_removed' in result_file: continue
        #if 'text_' not in result_file: continue
        print(result_file)
        try:
            with open(result_file, 'rb') as f:
                results = pickle.load(f)
            
            found = remove_coefs_from_results(results)

            if found:
                with open(result_file, 'wb') as f:
                    pickle.dump(results, f)
        except Exception as e:
            print('\tError: {}'.format(e))

if __name__ == '__main__':
    main()