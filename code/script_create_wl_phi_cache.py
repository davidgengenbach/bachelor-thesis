#!/usr/bin/env python

import pickle
import os
from joblib import delayed, Parallel

from kernels import fast_fast_wl
from utils import dataset_helper, graph_helper, graph_metrics, filename_utils, helper, time_utils
from utils.logger import LOGGER
from time import time


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='Create fast_wl phi cache')
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--wl_h', type=int, default=10)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--limit_dataset', nargs='+', type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    helper.print_script_args_and_info(args)
    Parallel(n_jobs=args.n_jobs)(delayed(process_graph_cache_file)(graph_cache_file, args) for graph_cache_file in dataset_helper.get_all_cached_graph_datasets())


def file_should_be_processed(cache_filename):
    not_wanted = ['.phi.', 'gram']
    for x in not_wanted:
        if x in cache_filename:
            return False
    return True

def process_graph_cache_file(graph_cache_file, args):
    graph_cache_filename = graph_cache_file.split('/')[-1].rsplit('.')[0]
    dataset = filename_utils.get_dataset_from_filename(graph_cache_file)

    if not file_should_be_processed(graph_cache_filename):
        return

    start = time()
    X, Y, adjs, labels = None, None, None, None

    node_weight_metrics = [None, graph_metrics.nxgraph_pagerank_metric, graph_metrics.nxgraph_degrees_metric]
    node_weight_metrics = [None, graph_metrics.nxgraph_degrees_metric]

    try:
        phi_graph_cache_file_ = graph_cache_file.replace('.npy', '.fast_wl.phi.npy')

        for metric in node_weight_metrics:
            for same_label in [True, False]:
                metric_name = metric.__name__ if metric else 'none'
                phi_graph_cache_file = phi_graph_cache_file_.replace('.npy', '.{}.{}.npy'.format(metric_name, 'same_labels' if same_label else 'original_labels'))

                if os.path.exists(phi_graph_cache_file):
                    continue

                # Lazy load dataset
                if not X:
                    X, Y = dataset_helper.get_dataset_cached(graph_cache_file)
                    X = graph_helper.get_graphs_only(X)
                    X_tuple = graph_helper.convert_graphs_to_adjs_tuples(X, copy=True)
                    adjs, labels = zip(*X_tuple)

                node_weight_factors = graph_metrics.get_node_weights_for_nxgraphs(X, metric)
                phi_list = fast_fast_wl.transform(adjs, labels, h=args.wl_h, node_weight_factors=node_weight_factors)
                with open(phi_graph_cache_file, 'wb') as f:
                    pickle.dump((phi_list, Y), f)
    except Exception as e:
        LOGGER.exception(e)
    LOGGER.info('{:15} finished (time={}) ({})'.format(dataset, time_utils.seconds_to_human_readable(time() - start),graph_cache_filename))


if __name__ == '__main__':
    main()
