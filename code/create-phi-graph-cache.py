#!/usr/bin/env python

import os
import dataset_helper
import graph_helper
import gc
import pickle
from transformers.fast_wl_graph_kernel_transformer import FastWLGraphKernelTransformer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

for graph_cache_file in dataset_helper.get_all_cached_graph_datasets():
    phi_graph_cache_file = graph_cache_file.replace('.npy', '.phi.npy')
    print('{} {}'.format('#' * 20, graph_cache_file.split('/')[-1]))
    if '.phi.' in graph_cache_file: continue
        
    if os.path.exists(phi_graph_cache_file):
        print('\tAlready calculated phi: {}'.format(phi_graph_cache_file))
        continue

    gc.collect()
    try:
        X, Y = dataset_helper.get_dataset_cached(graph_cache_file)
        transformer = FastWLGraphKernelTransformer(h=4, remove_missing_labels=True)
        transformer.fit(X)
        
        with open(phi_graph_cache_file, 'wb') as f:
            pickle.dump((transformer.phi_list, Y), f)
        del X, Y, transformer
    except Exception as e:
        logger.exception(e)
