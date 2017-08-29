#!/usr/bin/env python
from glob import glob
import pickle
from pprint import pprint
import os

for result_file in glob('data/results/*.npy'):
    if 'model_removed' in result_file: continue
    if 'text_' not in result_file: continue
    print(result_file)
    try:
        with open(result_file, 'rb') as f:
            results = pickle.load(f)
        print(results['param_clf'])
        if hasattr(results['param_clf'][0], 'coef_'):
            del results['param_clf'][0].coef_
            with open(result_file, 'wb') as f:
                pickle.dump(results, f)
    except Exception as e:
        print('\tError: {}'.format(e))

#os.makedirs('data/results/removed', exist_ok = True)
#for result_removed_file in glob('data/results/*model_removed*'):
#    old_path = "/".join(result_removed_file.split('/')[:-1])
#    
#    os.rename(result_removed_file, old_path + '/removed/' + result_removed_file.split('/')[-1])  
