#!/usr/bin/env python3
'''
Simple profiling test for fast_wl to find a bottleneck.

Install flamegraph as described on GitHub. So, not `pip install flamegraph`, but:
pip install git+https://github.com/evanhempel/python-flamegraph.git -U
'''

import os
import sys
import flamegraph

# Change dir to ..
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.join(dir_path, '..')
sys.path.append(parent_dir)
os.chdir(parent_dir)

from kernels import fast_wl
import scipy
import scipy.sparse
from time import time
from utils import dataset_helper, graph_helper, primes, time_utils


DATASET = 'ng20'
H = 10
USED_SPARSE_MATRIX = scipy.sparse.lil_matrix
PERF_LOG='./perf.log'

# Warm up primes cache
primes.get_highest_prime_range()

# Get dataset
X, Y = dataset_helper.get_concept_map_for_dataset(DATASET, graphs_only=True)

# Convert to adj
X = graph_helper.convert_graphs_to_adjs_tuples(X, copy=True)

# Same label for all nodes
adj, labels = zip(*X)
labels = [[1] * len(x) for x in labels]
X = list(zip(adj, labels))

# Start profiling
thread = flamegraph.start_profile_thread(fd=open(PERF_LOG, "w"))
start = time()

# Calculate fast_wl
phi_lists, new_label_lookups, new_label_counters = fast_wl.transform(X, h=H, used_matrix_type=USED_SPARSE_MATRIX)

# Convert to sparse lil_matrix for "fancy" indexing (needed for other pipeline elements)
phi_lists = [phi.tolil().T for phi in phi_lists]
print(time_utils.seconds_to_human_readable(time() - start))