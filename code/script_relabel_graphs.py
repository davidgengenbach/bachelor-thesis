#!/usr/bin/env python3

import os
import pickle

from joblib import Parallel, delayed

from transformers.nx_graph_to_tuple_transformer import NxGraphToTupleTransformer
from transformers.relabel_graphs_transformer import RelabelGraphsTransformer
from utils import dataset_helper


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Relabels graphs with a given lookup')
    parser.add_argument('--limit_dataset', type=str, default=None)
    parser.add_argument('--force', action = 'store_true')
    parser.add_argument('--lookup_path', type=str, default='data/embeddings/graph-embeddings')
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    Parallel(n_jobs=args.n_jobs)(delayed(process_dataset)(dataset_name, args) for dataset_name in dataset_helper.get_all_available_dataset_names())
    print('Finished')

def process_dataset(dataset, args):
    def print_dataset(msg = ''):
        print('{:14}{}'.format(dataset, ' - ' + msg if msg else ''))

    if args.limit_dataset and args.limit_dataset not in dataset: return
    print_dataset()
    tuple_trans = NxGraphToTupleTransformer()
    label_lookup_file = '{}/{}.label-lookup.npy'.format(args.lookup_path, dataset)
    if not os.path.exists(label_lookup_file):
        print_dataset('No label-lookup found for: {} ({})'.format(dataset, label_lookup_file))
    with open(label_lookup_file, 'rb') as f:
        label_lookup = pickle.load(f)
    
    print_dataset('Loaded lookup')
    relabel_trans = RelabelGraphsTransformer(label_lookup)
    print_dataset('Starting processing')
    for graph_dataset_cache_file in dataset_helper.get_all_cached_graph_datasets(dataset):
        result_file = graph_dataset_cache_file.replace(dataset, 'relabeled_{}'.format(dataset))

        if not args.force and os.path.exists(result_file):
            print_dataset('\t\tAlready processed, skipping: {}'.format(graph_dataset_cache_file))
            continue

        print_dataset('\tProcessing dataset: {}'.format(graph_dataset_cache_file))
        X, Y = dataset_helper.get_dataset_cached(graph_dataset_cache_file)

        X = tuple_trans.transform(X)
        print_dataset('\t\tRelabeling: {}'.format(graph_dataset_cache_file))
        X = relabel_trans.transform(X)
        print_dataset('\t\tSaving: {}, to {}'.format(graph_dataset_cache_file, result_file))
        with open(result_file, 'wb') as f:
            pickle.dump((X, Y), f)
        print_dataset('\t\tDone: {}'.format(graph_dataset_cache_file))

if __name__ == '__main__':
    main()