#!/usr/bin/env python

import os
import dataset_helper
import graph_helper
import gc
import pickle
from transformers.fast_wl_graph_kernel_transformer import FastWLGraphKernelTransformer
from joblib import delayed, Parallel

def main():
    args = get_args()
    Parallel(n_jobs=args.n_jobs)(delayed(process_dataset)(graph_cache_file, args) for graph_cache_file in dataset_helper.get_all_cached_graph_datasets())

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Create phi cache')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--h', type=int, default=4)
    parser.add_argument('--remove_missing_labels', type=bool, default=True)
    parser.add_argument('--force', action = 'store_true')
    parser.add_argument('--limit_dataset', type = str, default = None)
    args = parser.parse_args()
    return args

def process_dataset(graph_cache_file, args):
    if args.limit_dataset and args.limit_dataset not in graph_cache_file: return
    phi_graph_cache_file = graph_cache_file.replace('.npy', '.phi.npy')
    dataset_name = graph_cache_file.split('/')[-1]
    print('dataset: {:15}'.format(dataset_name))
    if '.phi.' in graph_cache_file: return

    if not args.force and os.path.exists(phi_graph_cache_file):
        print('dataset: {:15} - cache file already exists'.format(dataset_name))
        return
    gc.collect()

    with open(phi_graph_cache_file, 'w') as f:
        f.write('NOT_DONE')

    try:
        X, Y = dataset_helper.get_dataset_cached(graph_cache_file)
        transformer = FastWLGraphKernelTransformer(h=args.h, remove_missing_labels=args.remove_missing_labels)
        transformer.fit(X)
        
        with open(phi_graph_cache_file, 'wb') as f:
            pickle.dump((transformer.phi_list, Y), f)
        del X, Y, transformer
    except Exception as e:
        print(e)
    print('dataset: {:15} - finished'.format(dataset_name))


if __name__ == '__main__':
    main()