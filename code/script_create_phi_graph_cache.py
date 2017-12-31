#!/usr/bin/env python

import os
import pickle
from glob import glob
import sys

import numpy as np
from joblib import delayed, Parallel

from kernels import spgk
from transformers.fast_wl_graph_kernel_transformer import FastWLGraphKernelTransformer
from transformers.nx_graph_to_tuple_transformer import NxGraphToTupleTransformer
from utils import dataset_helper, graph_helper, filter_utils, filename_utils, helper
from utils.logger import LOGGER

import sklearn
from sklearn import utils


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Create phi cache')

    # General
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--force', action='store_true')

    parser.add_argument('--limit_dataset', nargs='+', type=str, default=None)
    parser.add_argument('--include_filter', type=str, default=None)
    parser.add_argument('--exclude_filter', type=str, default=None)

    # Kernel: WL
    parser.add_argument('--use_wl', type=helper.argparse_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--wl_h', type=int, default=10)
    parser.add_argument('--wl_sort_classes', action='store_true', default=True)
    parser.add_argument('--wl_test_size', type=float, default=0.2)

    # Kernel: SPGK
    parser.add_argument('--use_spgk', type=helper.argparse_str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--spgk_depth', nargs='+', type=int, default=[1, 2])
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    helper.print_script_args_and_info(args)
    Parallel(n_jobs=args.n_jobs)(delayed(process_graph_cache_file)(graph_cache_file, args) for graph_cache_file in dataset_helper.get_all_cached_graph_datasets())


def process_graph_cache_file(graph_cache_file, args):
    graph_cache_filename = graph_cache_file.split('/')[-1].rsplit('.')[0]
    dataset = filename_utils.get_dataset_from_filename(graph_cache_file)

    if '.phi.' in graph_cache_filename or not filter_utils.file_should_be_processed(graph_cache_filename, args.include_filter, args.exclude_filter, args.limit_dataset):
        return

    LOGGER.info('{:15} starting ({})'.format(dataset, graph_cache_filename))

    fast_wl_trans = FastWLGraphKernelTransformer(
        h=args.wl_h,
        use_early_stopping=False,
        truncate_to_highest_label=False
    )

    try:
        phi_graph_cache_file = graph_cache_file.replace('.npy', '.phi.npy')
        X_graphs, Y = dataset_helper.get_dataset_cached(graph_cache_file)
        X_graphs = graph_helper.get_graphs_only(X_graphs)

        # Kernel: WL
        if args.use_wl:
            used_phi_graph_cache_file = phi_graph_cache_file
            splitted_phi_graph_cache_file = phi_graph_cache_file.replace('.phi', '.splitted.phi')
            phi_same_label_graph_cache_file = phi_graph_cache_file.replace(dataset, '{}_same-label'.format(dataset)).replace('.phi', '.splitted.phi')

            # Stop here if all files have already been created
            if not args.force and np.all([os.path.exists(x) for x in [splitted_phi_graph_cache_file, used_phi_graph_cache_file, phi_same_label_graph_cache_file]]):
                return

            X_, Y_ = np.array(np.copy(X_graphs)), np.array(np.copy(Y))
            if args.wl_sort_classes:
                X_, Y_ = sort(X_, Y_, by=Y_)

            num_vertices = len(graph_helper.get_all_node_labels(X_))
            fast_wl_trans.set_params(phi_dim=num_vertices)

            X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(
                np.copy(X_),
                np.copy(Y_),
                stratify=Y_,
                test_size=args.wl_test_size
            )

            X_train, Y_train = sort(X_train, Y_train, by=Y_train)
            X_test, Y_test = sort(X_test, Y_test, by=Y_test)

            # Splitted version
            if args.force or not os.path.exists(splitted_phi_graph_cache_file):
                t = sklearn.base.clone(fast_wl_trans).set_params(same_label=True)
                phi_train = t.fit_transform(np.copy(X_train))
                phi_test = t.transform(np.copy(X_test))

                with open(splitted_phi_graph_cache_file, 'wb') as f:
                    pickle.dump((phi_train, phi_test, X_train, X_test, Y_train, Y_test), f)

            # Splitted, same label
            if args.force or not os.path.exists(phi_same_label_graph_cache_file):
                t = sklearn.base.clone(fast_wl_trans)
                phi_train = t.fit_transform(X_train)
                phi_test = t.transform(X_test)

                with open(phi_same_label_graph_cache_file, 'wb') as f:
                    pickle.dump((phi_train, phi_test, X_train, X_test, Y_train, Y_test), f)

            # Whole dataset
            if args.force or not os.path.exists(used_phi_graph_cache_file):
                t = sklearn.base.clone(fast_wl_trans)
                with open(used_phi_graph_cache_file, 'wb') as f:
                    pickle.dump((t.fit_transform(X_), Y_), f)

        # Kernel: spgk
        if args.use_spgk:
            for depth in args.spgk_depth:
                spgk_graph_cache_file = graph_cache_file.replace('.npy', '.spgk-{}.gram.npy'.format(depth))

                if args.force or not os.path.exists(spgk_graph_cache_file):
                    K = spgk.transform(X_graphs, depth=depth)

                    with open(spgk_graph_cache_file, 'wb') as f:
                        pickle.dump((K, Y), f)
    except Exception as e:
        LOGGER.exception(e)

    LOGGER.info('{:15} finished ({})'.format(dataset, graph_cache_filename))

def sort(*Xs, by=None):
    indices = np.argsort(by)
    return [np.array(x)[indices] for x in Xs]


if __name__ == '__main__':
    main()
