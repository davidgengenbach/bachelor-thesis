#!/usr/bin/env python3

import os
import pickle

from joblib import Parallel, delayed
from glob import glob
from time import time
from itertools import chain
import collections

import transformers
from utils import dataset_helper, filename_utils, helper, time_utils, constants, graph_helper

from utils.logger import LOGGER


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Relabels graphs with a given lookup')
    parser.add_argument('--limit_dataset', type=str, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--lookup_path', type=str, default='data/embeddings/graph-embeddings')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--max_occurrence', type=int, default=2)
    parser.add_argument('--limit_graphs', nargs='+', type=str, default=constants.TYPE_CONCEPT_MAP)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    helper.print_script_args_and_info(args)

    datasets = dataset_helper.get_all_available_dataset_names(limit_datasets=args.limit_dataset)
    graph_files = []
    for dataset in datasets:
        label_lookup_files = glob('{}/{}.*.label-lookup.npy'.format(args.lookup_path, dataset))
        for label_lookup_file in label_lookup_files:
            if not os.path.exists(label_lookup_file):
                print('No lookup file for dataset found: {}'.format(dataset))
                continue
            graph_files += [(cache_file, label_lookup_file) for cache_file in dataset_helper.get_all_cached_graph_datasets(dataset_name=dataset)]

    print('# Num tasks: {}'.format(len(graph_files)))

    Parallel(n_jobs=args.n_jobs)(delayed(process_dataset)(cache_file, label_lookup_file, args) for cache_file, label_lookup_file in graph_files)

    LOGGER.info('Finished')


def process_dataset(cache_file, label_lookup_file, args):
    start = time()
    dataset = filename_utils.get_dataset_from_filename(cache_file)

    cache_filename = filename_utils.get_filename_only(cache_file, with_extension=False)

    threshold, topn = filename_utils.get_topn_threshold_from_lookupfilename(label_lookup_file)

    result_file = cache_file.replace(dataset, 'relabeled_threshold_{}_topn_{}_{}'.format(threshold, topn, dataset))
    if not args.force and os.path.exists(result_file):
        return

    LOGGER.info('{:80} topn={:4} threshold={:4} Starting'.format(cache_filename , topn, threshold))

    with open(label_lookup_file, 'rb') as f:
        label_lookup = pickle.load(f)

    X, Y = dataset_helper.get_dataset_cached(cache_file)
    X = graph_helper.get_graphs_only(X)

    # Get label to be renamed
    node_labels = list(chain.from_iterable([x.nodes() for x in X]))
    counter = collections.Counter(node_labels)

    node_labels_to_be_renamed = set([label for label, occurrences in counter.items() if occurrences <= args.max_occurrence])

    lookup_ = {
        label: new_label for label, new_label in label_lookup.items() if label in node_labels_to_be_renamed
    }

    new_labels = set(lookup_.values())
    lookup__ = collections.defaultdict(list)

    for label, new_label in label_lookup.items():
        if new_label in new_labels:
            lookup__[label].append(new_label)

    lookup_ = dict(lookup_, **lookup__)

    relabel_trans = transformers.RelabelGraphsTransformer(lookup_)

    X = relabel_trans.transform(X)

    with open(result_file, 'wb') as f:
        pickle.dump((X, Y), f)

    time_needed = time() - start

    LOGGER.info('{:80} topn={:4} threshold={:4} Finished (time={})'.format(cache_filename, topn, threshold, time_utils.seconds_to_human_readable(time_needed)))

if __name__ == '__main__':
    main()
