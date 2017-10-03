"""Implementation of the fast hashing-based algorithm.
Ported from the MATLAB version: https://github.com/rmgarnett/fast_wl

Attributes:
    primes_arguments_required (list):
"""
import numpy as np
import scipy
from scipy.sparse import lil_matrix, dok_matrix
from utils import primes
import collections
import typing

# https://oeis.org/A033844
primes_arguments_required_ = [2, 3, 7, 19, 53, 131, 311, 719, 1619, 3671, 8161, 17863, 38873, 84017, 180503, 386093, 821641, 1742537, 3681131, 7754077, 16290047, 34136029, 71378569, 148948139, 310248241, 645155197, 1339484197, 2777105129, 5750079047, 11891268401, 24563311309, 50685770167, 104484802057, 215187847711]


def transform(
        graphs: typing.List[typing.Tuple[scipy.sparse.spmatrix, typing.Iterable]],
        h: int = 1,
        label_lookups: typing.List[typing.Dict] = None,
        label_counters: typing.List[int] = None,
        primes_arguments_required: typing.List[int] = primes_arguments_required_,
        phi_dim: typing.Tuple = None,
        labels_dtype: np.dtype = np.uint32,
        phi_dtype: np.dtype = np.uint32,
        used_matrix_type: scipy.sparse.spmatrix = dok_matrix,
        round_signatures_to_decimals: int = 10,
        cast_after_rounding: bool = False
    ) -> tuple:

    assert len(graphs)

    # If no previous label counters/lookups are given, create empty ones
    if not label_lookups:
        label_lookups = [{} for x in range(h + 1)]
    if not label_counters:
        label_counters = [0] * (h + 1)

    adjacency_matrices = [adjs for adjs, nodes in graphs]

    # Remove weight information (not needed for this WL kernel)
    for idx, adj in enumerate(adjacency_matrices):
        adj = adj.tocsr()
        adj.data = np.where(adj.data > 1, 0, 1)
        adjacency_matrices[idx] = adj
        #adj[adj.nonzero()] = 1

    # Relabel the graphs, mapping the string labels to unique IDs (ints)
    label_lookup, label_counter, graph_labels = relabel_graphs(graphs, label_counter = label_counters[0], label_lookup = label_lookups[0], labels_dtype = labels_dtype)

    num_labels = len(label_lookup.keys())
    num_graphs = len(graphs)

    assert len(graph_labels) == len(graphs)

    # Save the label_lookups/label_counters for later use
    new_label_lookups = [label_lookup]
    new_label_counters = [label_counter]

    # The upper bound up to which the prime numbers have to be retrieved
    primes_needed = primes_arguments_required[int(np.ceil(np.log2(num_labels)) + 3) + 3]
    log_primes = primes.get_log_primes(1, primes_needed)

    # The number of unique labels (= the total number of nodes in graphs)
    if not phi_dim:
        phi_dim = sum(len(x) for x in graph_labels)

    phi_shape = (phi_dim, num_graphs)

    # Just count the labels in the 0-th iteration. This corresponds to the graph_labels, but indexed correctly into phi
    phi = used_matrix_type(phi_shape, dtype=phi_dtype)
    for idx, labels in enumerate(graph_labels):
        phi[labels, idx] = 1

    phi_lists = [phi.tolil()]

    # For the number of iterations h...
    for i in range(h):
        # ... use previous label counters/lookups, if given
        label_lookup = label_lookups[i + 1]
        label_counter = label_counters[i + 1]

        assert isinstance(label_counter, int)
        assert isinstance(label_lookup, dict)

        phi = used_matrix_type(phi_shape, dtype=phi_dtype)
        # ... go over all graphs
        for idx, (labels, adjacency_matrix) in enumerate(zip(graph_labels, adjacency_matrices)):
            has_same_labels = len(set(labels)) != len(labels)


            # ... generate the signatures (see paper) for each graph
            signatures = np.round((labels + adjacency_matrix * log_primes[labels]), decimals= round_signatures_to_decimals)

            if cast_after_rounding:
                signatures = signatures.astype(labels_dtype)

            # ... add missing signatures to the label lookup
            for signature in signatures:
                if signature not in label_lookup:
                    label_lookup[signature] = label_counter
                    label_counter += 1

            # ... relabel the graphs with the new (compressed) labels
            new_labels = np.array([label_lookup[signature] for signature in signatures], dtype = labels_dtype)
            graph_labels[idx] = new_labels

            # ... increment the entries in phi by one, where a label is given
            if has_same_labels:
                # Increment by one. Unfortunately you can not just use np.add.at(...) for duplicate indices to be accumulated
                label_counter_ = collections.Counter(new_labels)
                # new_label_indices: are the unique (!) indices in new_labels
                # vals: are the number of occurrences of a index in new_labels
                new_label_indices, vals = zip(*label_counter_.items())
                phi[list(new_label_indices), idx] += np.matrix(list(vals), dtype = phi.dtype).T
            else:
                # Set to one. This is way faster!
                phi[new_labels, idx] = 1
        # ... save phi
        phi_lists.append(phi.tolil())
        # ... save label counters/lookups for later use
        new_label_counters.append(label_counter)
        new_label_lookups.append(label_lookup)
    # Return the phis, the lookups and counter, so the calculation can resumed later on with new graphs
    return phi_lists, new_label_lookups, new_label_counters


def relabel_graphs(graphs: collections.abc.Iterable, label_counter: int = 0, label_lookup: dict = {}, labels_dtype: np.dtype = np.uint32, append: bool = True):
    assert isinstance(label_counter, int)
    assert isinstance(label_lookup, dict)

    labels = [[] for i in range(len(graphs))]
    nodes = [nodes for adjs, nodes in graphs]
    for idx, nodes_ in enumerate(nodes):
        for label in nodes_:
            if append and label not in label_lookup:
                label_lookup[label] = label_counter
                label_counter += 1
        labels[idx] = np.array([label_lookup[x] for x in nodes_], dtype=labels_dtype)
    return label_lookup, label_counter, labels
