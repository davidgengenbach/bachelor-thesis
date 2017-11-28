#!/usr/bin/env python3

import os
import pickle

from joblib import Parallel, delayed

from transformers.nx_graph_to_tuple_transformer import NxGraphToTupleTransformer
from transformers.relabel_graphs_transformer import RelabelGraphsTransformer
from utils import dataset_helper, filename_utils, time_utils

from utils.logger import LOGGER


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Relabels graphs with a given lookup')
    parser.add_argument('--limit_dataset', type=str, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--lookup_path', type=str, default='data/embeddings/graph-embeddings')
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    datasets = dataset_helper.get_all_available_dataset_names(args.limit_dataset)

    graph_files = []
    for dataset in datasets:
        label_lookup_file = '{}/{}.label-lookup.npy'.format(args.lookup_path, dataset)
        if not os.path.exists(label_lookup_file):
            print('No lookup file for dataset found: {}'.format(dataset))
            continue
        graph_files += [(cache_file, label_lookup_file) for cache_file in dataset_helper.get_all_cached_graph_datasets(dataset_name=dataset)]

    Parallel(n_jobs=args.n_jobs)(delayed(process_dataset)(cache_file, label_lookup_file, args) for cache_file, label_lookup_file in graph_files)

    LOGGER.info('Finished')


def process_dataset(cache_file, label_lookup_file, args):
    dataset = filename_utils.get_dataset_from_filename(cache_file)

    result_file = cache_file.replace(dataset, 'relabeled_{}'.format(dataset))
    if not args.force and os.path.exists(result_file):
        return

    with open(label_lookup_file, 'rb') as f:
        label_lookup = pickle.load(f)

    tuple_trans = NxGraphToTupleTransformer()
    relabel_trans = RelabelGraphsTransformer(label_lookup)

    X, Y = dataset_helper.get_dataset_cached(cache_file)
    X = tuple_trans.transform(X)
    X = relabel_trans.transform(X)
    with open(result_file, 'wb') as f:
        pickle.dump((X, Y), f)


if __name__ == '__main__':
    main()
