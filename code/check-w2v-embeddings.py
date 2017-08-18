#!/usr/bin/env python

import dataset_helper
import graph_helper

def main():
    for dataset_name in dataset_helper.get_all_available_dataset_names():
        print('Processing: {}'.format(dataset_name))
        model = dataset_helper.get_w2v_embedding_for_dataset(dataset_name)
        graph_cache_files = [x for x in dataset_helper.get_all_cached_graph_datasets() if dataset_name in x]

        if len(graph_cache_files) <= 2:
            print('\tNo graph cache files found'.format(dataset_name))
            continue
        for graph_cache_file in graph_cache_files[:2]:
            print('\t{}'.format(graph_cache_file))
            X, Y = dataset_helper.get_dataset_cached(graph_cache_file)
            labels = graph_helper.get_all_node_labels(X)
            print('\t#unique labels: {}'.format(len(labels)))
            counter = {'found': 0, 'not_found': 0}
            for idx, label in enumerate(labels):
                if label in model:
                    counter['found'] += 1
                else:
                    counter['not_found'] += 1
            print('\t{}, Found: {}%'.format(counter, int(100 * counter['found'] / len(labels))))

if __name__ == '__main__':
    main()