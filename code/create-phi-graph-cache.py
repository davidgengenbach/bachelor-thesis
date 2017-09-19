#!/usr/bin/env python

import os
import dataset_helper
import graph_helper
import coreference
import gc
import pickle
from transformers.fast_wl_graph_kernel_transformer import FastWLGraphKernelTransformer
from joblib import delayed, Parallel
from transformers.nx_graph_to_tuple_transformer import NxGraphToTupleTransformer
from transformers.relabel_graphs_transformer import RelabelGraphsTransformer
from logger import LOGGER
from glob import glob
import sys
from kernels import spgk
import numpy as np

import re

def main():
    args = get_args()
    Parallel(n_jobs=args.n_jobs)(delayed(process_dataset)(graph_cache_file, args) for graph_cache_file in dataset_helper.get_all_cached_graph_datasets())


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Create phi cache')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--h', type=int, default=4)
    parser.add_argument('--remove_missing_labels', type=bool, default=True)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--disable_wl', action='store_true')
    parser.add_argument('--disable_spgk', action='store_true')
    parser.add_argument('--limit_dataset', nargs='+', type=str, default=['ng20', 'ling-spam', 'reuters-21578', 'webkb'], dest='limit_dataset')
    parser.add_argument('--spgk_depth', nargs='+', type=int, default=[1])
    parser.add_argument('--lookup_path', type=str, default='data/embeddings/graph-embeddings')
    args = parser.parse_args()
    return args


def process_dataset(graph_cache_file, args):
    graph_cache_filename = graph_cache_file.split('/')[-1].rsplit('.')[0]
    dataset = dataset_helper.get_dataset_name_from_graph_cachefile(graph_cache_file)

    if args.limit_dataset and dataset not in args.limit_dataset:
        return

    label_lookup_files = glob('{}/{}.*.label-lookup.npy'.format(args.lookup_path, dataset))

    tuple_trans = NxGraphToTupleTransformer()
    fast_wl_trans = FastWLGraphKernelTransformer(h=args.h, remove_missing_labels=args.remove_missing_labels)

    if '.phi.' in graph_cache_file:
        return

    LOGGER.info('{:15}'.format(dataset))

    gc.collect()

    try:
        X_graphs, Y = dataset_helper.get_dataset_cached(graph_cache_file)
        X = tuple_trans.transform(np.copy(X_graphs))
        phi_graph_cache_file = graph_cache_file.replace('.npy', '.phi.npy')
        phi_same_label_graph_cache_file = phi_graph_cache_file.replace(dataset, 'same-label_{}'.format(dataset))

        if not args.disable_wl:
            # Without relabeling
            if args.force or not os.path.exists(phi_graph_cache_file):
                LOGGER.info('{:15} \tProcessing: {}'.format(dataset, phi_graph_cache_file))
                fast_wl_trans.fit(X)
                with open(phi_graph_cache_file, 'wb') as f:
                    pickle.dump((fast_wl_trans.phi_list, Y), f)
                LOGGER.info('{:15} \tDone: {}'.format(dataset, phi_graph_cache_file))

            # All nodes get same label
            if args.force or not os.path.exists(phi_same_label_graph_cache_file):
                LOGGER.info('{:15} \tProcessing: {}'.format(dataset, phi_same_label_graph_cache_file))
                X_same_label = [(x, [0] * len(y)) for x, y in X]
                fast_wl_trans.fit(X_same_label)
                with open(phi_same_label_graph_cache_file, 'wb') as f:
                    pickle.dump((fast_wl_trans.phi_list, Y), f)
                LOGGER.info('{:15} \tDone: {}'.format(dataset, phi_same_label_graph_cache_file))

            # With relabeling
            for label_lookup_file in label_lookup_files:
                threshold = '.'.join(label_lookup_file.split('threshold-')[1].split('.')[:2])
                topn = label_lookup_file.split('topn-')[1].split('.')[0]
                
                phi_graph_relabeled_cache_file = phi_graph_cache_file.replace(dataset, 'relabeled_threshold_topn-{}_threshold-{}_{}'.format(topn, threshold, dataset))

                if args.force or not os.path.exists(phi_graph_relabeled_cache_file):
                    with open(label_lookup_file, 'rb') as f:
                        label_lookup = pickle.load(f)

                    LOGGER.info('{:15} \tProcessing: {}, threshold: {}, topn: {}'.format(dataset, phi_graph_relabeled_cache_file, threshold, topn))
                    LOGGER.info('{:15} \tProcessing: {}, relabeling'.format(dataset, phi_graph_relabeled_cache_file))
                    relabel_trans = RelabelGraphsTransformer(label_lookup)

                    X = relabel_trans.transform(X)

                    LOGGER.info('{:15} \tProcessing: {}, fixing duplicate labels'.format(dataset, phi_graph_relabeled_cache_file))
                    X = [coreference.fix_duplicate_label_graph(*x) for x in X]

                    LOGGER.info('{:15} \tProcessing: {}, fast_wl'.format(dataset, phi_graph_relabeled_cache_file))
                    fast_wl_trans.fit(X)
                    
                    with open(phi_graph_relabeled_cache_file, 'wb') as f:
                        pickle.dump((fast_wl_trans.phi_list, Y), f)

                    LOGGER.info('{:15} \tDone: {}'.format(dataset, phi_graph_relabeled_cache_file))

        if not args.disable_spgk:
            for depth in args.spgk_depth:
                spgk_graph_cache_file =graph_cache_file.replace('.npy', '.spgk-{}.gram.npy'.format(depth))

                if args.force or not os.path.exists(spgk_graph_cache_file):
                    LOGGER.info('{:15} \tProcessing: {}'.format(dataset, spgk_graph_cache_file))

                    X_new = np.copy(X_graphs)
                    # "Repair" graph (remove self-loops and set weight of all edges to 1)
                    for x in X_new:
                        for u,v,edata in x.edges(data = True):
                            if 'weight' not in edata: continue
                            edata['weight'] = 1
                        self_loop_edges = x.selfloop_edges()
                        if len(self_loop_edges):
                            x.remove_edges_from(self_loop_edges)

                    K = spgk.build_kernel_matrix(X_new, depth = depth)
                    with open(spgk_graph_cache_file, 'wb') as f:
                        pickle.dump((K, Y), f)
                    LOGGER.info('{:15} \tDone: {}'.format(dataset, spgk_graph_cache_file))

        del X, Y, fast_wl_trans
    except Exception as e:
       LOGGER.exception(e)
    LOGGER.info('{:15} finished'.format(dataset))


if __name__ == '__main__':
    main()
