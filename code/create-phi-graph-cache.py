#!/usr/bin/env python

import os
import dataset_helper
import graph_helper
import gc
import pickle
from transformers.fast_wl_graph_kernel_transformer import FastWLGraphKernelTransformer
from joblib import delayed, Parallel
from transformers.nx_graph_to_tuple_transformer import NxGraphToTupleTransformer
from transformers.relabel_graphs_transformer import RelabelGraphsTransformer


def main():
    args = get_args()
    Parallel(n_jobs=args.n_jobs)(delayed(process_dataset)(graph_cache_file, args)
                                 for graph_cache_file in dataset_helper.get_all_cached_graph_datasets())


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Create phi cache')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--h', type=int, default=1)
    parser.add_argument('--remove_missing_labels', type=bool, default=True)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--limit_dataset', type=str, default=None)
    parser.add_argument('--lookup_path', type=str, default='data/embeddings/graph-embeddings')
    args = parser.parse_args()
    return args


def process_dataset(graph_cache_file, args):
    graph_cache_filename = graph_cache_file.split('/')[-1].rsplit('.')[0]
    if args.limit_dataset and args.limit_dataset not in graph_cache_file:
        return

    dataset = dataset_helper.get_dataset_name_from_graph_cachefile(graph_cache_file)

    clean_dataset_name = dataset.replace('-single', '')
    label_lookup_file = '{}/{}.label-lookup.npy'.format(args.lookup_path, clean_dataset_name)

    with open(label_lookup_file, 'rb') as f:
        label_lookup = pickle.load(f)

    tuple_trans = NxGraphToTupleTransformer()
    relabel_trans = RelabelGraphsTransformer(label_lookup)
    fast_wl_trans = FastWLGraphKernelTransformer(h=args.h, remove_missing_labels=args.remove_missing_labels)

    phi_graph_cache_file = graph_cache_file.replace('.npy', '.phi.npy')
    if '.phi.' in graph_cache_file:
        return
    print('{:15}'.format(dataset))

    gc.collect()

    try:
        X, Y = dataset_helper.get_dataset_cached(graph_cache_file)
        X = tuple_trans.transform(X)

        # Without relabeling
        if args.force or not os.path.exists(phi_graph_cache_file):
            print('{:15} \tProcessing: {}'.format(dataset, phi_graph_cache_file))
            fast_wl_trans.fit(X)
            with open(phi_graph_cache_file, 'wb') as f:
                pickle.dump((fast_wl_trans.phi_list, Y), f)
            print('{:15} \tDone: {}'.format(dataset, phi_graph_cache_file))

        # With relabeling
        phi_graph_relabeled_cache_file = phi_graph_cache_file.replace(dataset, 'relabeled_{}'.format(dataset))
        if args.force or not os.path.exists(phi_graph_relabeled_cache_file):
            print('{:15} \tProcessing: {}'.format(dataset, phi_graph_relabeled_cache_file))
            X = relabel_trans.transform(X)

            fast_wl_trans.fit(X)
            with open(phi_graph_relabeled_cache_file, 'wb') as f:
                pickle.dump((fast_wl_trans.phi_list, Y), f)
            print('{:15} \tDone: {}'.format(dataset, phi_graph_relabeled_cache_file))

        del X, Y, fast_wl_trans
    except Exception as e:
       print(e)
    print('{:15} finished'.format(dataset))


if __name__ == '__main__':
    main()
