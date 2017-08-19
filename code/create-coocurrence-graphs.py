#!/usr/bin/env python

import dataset_helper
import graph_helper
import os
from joblib import delayed, Parallel

def main():
    args = get_args()
    print('args=', args)
    Parallel(n_jobs=args.n_jobs)(delayed(process_dataset)(dataset_name, args) for dataset_name in dataset_helper.get_all_available_dataset_names())

def process_dataset(dataset, args):
    print('dataset: {:15}'.format(dataset))
    for window_size in range(args.window_size_start, args.window_size_end):
        for only_nouns in [True, False]:
            print('dataset: {:15} - window_size={}, only_nouns={}'.format(dataset, window_size,  only_nouns))
            cache_file = dataset_helper.CACHE_PATH + '/dataset_graph_cooccurrence_{}_{}_{}.npy'.format(window_size, 'only-nouns' if only_nouns else 'all', dataset)
            print('dataset: {:15} - writing to cache file: {}'.format(dataset, cache_file))
            if not args.force and os.path.exists(cache_file):
                print('dataset: {:15} - cache file exists'.format(dataset, cache_file))
                continue
            with open(cache_file, 'w') as f:
                f.write('NOT_DONE')

            def process(X, Y):
                return graph_helper.convert_dataset_to_co_occurence_graph_dataset(X, Y, only_nouns = only_nouns, min_length = args.min_length, window_size = window_size, n_jobs = args.n_jobs_coo)

            dataset_helper.get_dataset(dataset, preprocessed = False, use_cached=False, transform_fn=process, cache_file=cache_file)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Calculates the cooccurrence matrices and saves them')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--n_jobs_coo', type=int, default=1)
    parser.add_argument('--window_size_start', type=int, default=1)
    parser.add_argument('--window_size_end', type=int, default=4)
    parser.add_argument('--min_length', type=int, default=-1)
    parser.add_argument('--force', action = 'store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()