#!/usr/bin/env python3

import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(dir_path, '..')
sys.path.append(parent_dir)
os.chdir(parent_dir)

from glob import glob
from utils import results_helper, constants
from utils.remove_coefs_from_results import remove_coefs_from_results
import pickle
import tqdm

for folder in [constants.RESULTS_FOLDER, results_helper.get_result_folders()[-1]]:
    result_files = glob('{}/*.npy'.format(folder))

    print('Processing folder: {:50} ({} items)'.format(folder, len(result_files)))
    for file in tqdm.tqdm(result_files):
        try:
            with open(file, 'rb') as f:
                res = pickle.load(f)
            found = remove_coefs_from_results(res['results'])
            if found:
                with open(file, 'wb') as f:
                    pickle.dump(res, f)
        except Exception as e:
            print('Error removing coefs: {} (message: {})'.format(file, e))