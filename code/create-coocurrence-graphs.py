#!/usr/bin/env python

import dataset_helper
import graph_helper
import os

for dataset in dataset_helper.get_all_available_dataset_names():
    min_length = -1
    print('{} Processing dataset: {}'.format('#' * 10, dataset))
    for window_size in range(1, 4):
        for only_nouns in [True, False]:
            def process(X, Y):
                return graph_helper.convert_dataset_to_co_occurence_graph_dataset(X, Y, only_nouns = only_nouns, min_length = min_length, window_size = window_size, n_jobs = 1)
            print("Creating co-occurence graphs for: {}".format(dataset))
            cache_file = dataset_helper.CACHE_PATH + '/dataset_graph_cooccurrence_{}_{}_{}.npy'.format(window_size, 'only-nouns' if only_nouns else 'all', dataset)
            if os.path.exists(cache_file):
                print('Cachefile already exists.')
                continue
            print('Cache file: {}'.format(cache_file))
            with open(cache_file, 'w') as f:
                f.write('TODO!')
            X, Y = dataset_helper.get_dataset(dataset, preprocessed = False, use_cached=False, transform_fn=process, cache_file=cache_file)