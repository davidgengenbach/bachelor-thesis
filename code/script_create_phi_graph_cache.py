#!/usr/bin/env python

import os
import pickle
from glob import glob
import sys

import numpy as np
from joblib import delayed, Parallel

from kernels import simple_set_matching
from kernels import spgk
from relabeling import coreference
from transformers.fast_wl_graph_kernel_transformer import FastWLGraphKernelTransformer
from transformers.nx_graph_to_tuple_transformer import NxGraphToTupleTransformer
from transformers.relabel_graphs_transformer import RelabelGraphsTransformer
from utils import dataset_helper, graph_helper, filter_utils, filename_utils, helper
from classification.classification_tasks import Task
from utils.logger import LOGGER

import sklearn
from sklearn import utils

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Create phi cache')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--wl_h', type=int, default=6)
    parser.add_argument('--max_path_distance_to_add', type=int, default=2)
    parser.add_argument('--remove_missing_labels', type=bool, default=True)
    parser.add_argument('--wl_sort_classes', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--disable_wl', action='store_true')
    parser.add_argument('--disable_spgk', action='store_true')
    parser.add_argument('--disable_simple_set_matching_kernel', action='store_true')
    parser.add_argument('--disable_relabeling', action='store_true')
    parser.add_argument('--limit_dataset', nargs='+', type=str, default=['ng20', 'ling-spam', 'reuters-21578', 'webkb'], dest='limit_dataset')
    parser.add_argument('--spgk_depth', nargs='+', type=int, default=[1])
    parser.add_argument('--lookup_path', type=str, default='data/embeddings/graph-embeddings')
    parser.add_argument('--include_filter', type=str, default=None)
    parser.add_argument('--exclude_filter', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    helper.print_script_args_and_info(args)
    Parallel(n_jobs=args.n_jobs)(delayed(process_graph_cache_file)(graph_cache_file, args) for graph_cache_file in dataset_helper.get_all_cached_graph_datasets())


def sort(*Xs, by):
    indices = np.argsort(by)
    return [np.array(x)[indices] for x in Xs]


def process_graph_cache_file(graph_cache_file, args):
    graph_cache_filename = graph_cache_file.split('/')[-1].rsplit('.')[0]
    dataset = filename_utils.get_dataset_from_filename(graph_cache_file)

    if '.phi.' in graph_cache_filename or not filter_utils.file_should_be_processed(graph_cache_filename, args.include_filter, args.exclude_filter, args.limit_dataset):
        return

    LOGGER.info('{:15} starting ({})'.format(dataset, graph_cache_filename))

    label_lookup_files = glob('{}/{}.*.label-lookup.npy'.format(args.lookup_path, dataset))

    tuple_trans = NxGraphToTupleTransformer()
    fast_wl_trans = FastWLGraphKernelTransformer(h=args.wl_h, remove_missing_labels=args.remove_missing_labels)

    try:
        phi_graph_cache_file = graph_cache_file.replace('.npy', '.phi.npy')
        phi_same_label_graph_cache_file = phi_graph_cache_file.replace(dataset, '{}_same-label'.format(dataset))

        X_graphs, Y = dataset_helper.get_dataset_cached(graph_cache_file)

        #X_graphs_walk_added = copy.deepcopy(X_graphs)

        # for graph in X_graphs_walk_added:
        #    graph_helper.add_shortest_path_edges(graph, cutoff = args.max_path_distance_to_add)

        if not args.disable_wl:
            X_, Y_ = np.array(np.copy(X_graphs)), np.array(np.copy(Y))

            if args.wl_sort_classes:
                X_, Y_ = sort(X_, Y_, by = Y_)

            X_ = tuple_trans.transform(X_)
            # Without relabeling
            used_phi_graph_cache_file = phi_graph_cache_file

            splitted_phi_graph_cache_file = phi_graph_cache_file.replace('.phi', '.splitted.phi')

            if args.force or not os.path.exists(splitted_phi_graph_cache_file):
                num_vertices = sum([len(labels) for _, labels in X_])
                test_size = 0.2
                random_state = 42
                X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(np.copy(X_), np.copy(Y_), stratify=Y_, test_size=test_size, random_state=random_state)

                fast_wl_trans.set_params(phi_dim = num_vertices)

                X_train, Y_train = sort(X_train, Y_train, by = Y_train)
                X_test, Y_test = sort(X_test, Y_test, by = Y_test)

                fast_wl_trans.fit(X_train)
                phi_train = fast_wl_trans.phi_list
                phi_test = fast_wl_trans.transform(X_test)

                with open(splitted_phi_graph_cache_file , 'wb') as f:
                    pickle.dump((phi_train, phi_test, X_train, X_test, Y_train, Y_test), f)


            if args.force or not os.path.exists(used_phi_graph_cache_file):
                fast_wl_trans.fit(X_)
                with open(used_phi_graph_cache_file , 'wb') as f:
                    pickle.dump((fast_wl_trans.phi_list, Y_), f)

            fast_wl_trans.set_params(should_cast=False)

            # All nodes get same label
            if args.force or not os.path.exists(phi_same_label_graph_cache_file):
                X_same_label = [(x, [0] * len(y)) for x, y in X_]
                fast_wl_trans.fit(X_same_label)
                with open(phi_same_label_graph_cache_file, 'wb') as f:
                    pickle.dump((fast_wl_trans.phi_list, Y_), f)

            # With relabeling
            if not args.disable_relabeling:
                for label_lookup_file in label_lookup_files:
                    threshold = '.'.join(label_lookup_file.split('threshold-')[1].split('.')[:2])
                    topn = label_lookup_file.split('topn-')[1].split('.')[0]

                    phi_graph_relabeled_cache_file = phi_graph_cache_file.replace(dataset, 'relabeled_threshold_topn-{}_threshold-{}_{}'.format(topn, threshold, dataset))

                    if args.force or not os.path.exists(phi_graph_relabeled_cache_file):
                        with open(label_lookup_file, 'rb') as f:
                            label_lookup = pickle.load(f)

                        relabel_trans = RelabelGraphsTransformer(label_lookup)

                        X_ = relabel_trans.transform(X_)
                        X_ = [coreference.fix_duplicate_label_graph(*x) for x in X_]

                        fast_wl_trans.fit(X_)

                        with open(phi_graph_relabeled_cache_file, 'wb') as f:
                            pickle.dump((fast_wl_trans.phi_list, Y_), f)

        if not args.disable_simple_set_matching_kernel:
            simple_kernel_cache_file = graph_cache_file.replace('.npy', '.simple.gram.npy')
            if args.force or not os.path.exists(simple_kernel_cache_file):
                K = simple_set_matching.transform(X_graphs)
                with open(simple_kernel_cache_file, 'wb') as f:
                    pickle.dump((K, Y), f)

        if not args.disable_spgk:
            for depth in args.spgk_depth:
                spgk_graph_cache_file = graph_cache_file.replace('.npy', '.spgk-{}.gram.npy'.format(depth))

                if args.force or not os.path.exists(spgk_graph_cache_file):
                    X_new = np.copy(X_graphs)
                    # "Repair" graph (remove self-loops and set weight of all edges to 1)
                    for x in X_new:
                        for u, v, edata in x.edges(data=True):
                            if 'weight' not in edata:
                                continue
                            edata['weight'] = 1
                        self_loop_edges = x.selfloop_edges()
                        if len(self_loop_edges):
                            x.remove_edges_from(self_loop_edges)

                    K = spgk.transform(X_new, depth=depth)

                    with open(spgk_graph_cache_file, 'wb') as f:
                        pickle.dump((K, Y), f)
    except Exception as e:
        LOGGER.exception(e)
    LOGGER.info('{:15} finished ({})'.format(dataset, graph_cache_filename))


if __name__ == '__main__':
    main()
