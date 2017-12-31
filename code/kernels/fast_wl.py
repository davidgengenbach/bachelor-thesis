"""Implementation of the fast hashing-based Weißfeiler Lehman algorithm.
Ported from the MATLAB version: https://github.com/rmgarnett/fast_wl
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
        phi_dim: int = None,
        labels_dtype: np.dtype = np.uint32,
        phi_dtype: np.dtype = np.uint32,
        used_matrix_type: scipy.sparse.spmatrix = lil_matrix,
        round_signatures_to_decimals: int = 1,
        append_to_labels: bool = True,
        ignore_label_order = False,
        node_weight_factors = None,
        use_early_stopping = True,
        fill_up_missing_iterations = False,
        node_weight_iteration_weight_function=None,
        truncate_to_highest_label=False
    ) -> tuple:
    assert len(graphs)

    # If no previous label counters/lookups are given, create empty ones
    if label_lookups is None:
        label_lookups = [{} for x in range(h + 1)]

    if label_counters is None:
        label_counters = [0] * (h + 1)

    adjacency_matrices = [adjs for adjs, nodes in graphs]

    # Remove weight information (not needed for this WL kernel)
    for idx, adj in enumerate(adjacency_matrices):
        if ignore_label_order:
            adj = adj.tolil()
            adj_dim = list(range(adj.shape[0]))
            adj[adj_dim, adj_dim] = 1
        adj = adj.tocsr()
        adj.data = np.where(adj.data < 1, 0, 1)
        adjacency_matrices[idx] = adj
    # Relabel the graphs, mapping the string labels to unique IDs (ints)
    label_lookup, label_counter, graph_labels = relabel_graphs(graphs, label_counter = label_counters[0], label_lookup = label_lookups[0], labels_dtype = labels_dtype, append = append_to_labels)
    # Save the label_lookups/label_counters for later use
    new_label_lookups = [label_lookup]
    new_label_counters = [label_counter]

    num_graphs = len(graphs)
    num_vertices = sum(len(x) for x in graph_labels)
    phi_width = num_vertices * 5

    assert len(graph_labels) == len(graphs)
    assert node_weight_factors is None or len(node_weight_factors) == len(graphs)

    # The upper bound up to which the prime numbers have to be retrieved
    primes_needed = primes_arguments_required[int(np.ceil(np.log2(phi_width))) + 1]
    log_primes = primes.get_log_primes(1, primes_needed)

    if node_weight_factors is not None:
        node_weight_factors = [np.array(x, dtype=object) for x in node_weight_factors]

    def get_phi_dim(iteration):
        if phi_dim is None: return None
        if isinstance(phi_dim, int):
            return phi_dim
        return phi_dim[iteration]

    def add_labels_to_phi_(labels_, iteration):
        data = []
        row_ind = []
        col_ind = []
        for graph_idx, labels in enumerate(labels_):
            if node_weight_factors is not None:
                factor = node_weight_factors[graph_idx]
            else:
                factor = 1

            if node_weight_iteration_weight_function:
                factor *= node_weight_iteration_weight_function(iteration)

            num_labels = len(labels)
            if isinstance(factor, (int, float)):
                data += [factor] * num_labels
            else:
                data += list(factor)

            row_ind += [graph_idx] * num_labels
            col_ind += list(labels)

        data, row_ind, col_ind = np.array(data), np.array(row_ind), np.array(col_ind)

        highest_label = np.max(col_ind) + 1
        phi_dim_ = get_phi_dim(iteration + 1)

        if truncate_to_highest_label:
            if phi_dim is None:
                phi_width_ = highest_label
            else:
                phi_width_ = phi_dim_

                idxs = np.where(col_ind < phi_width_)
                data = data[idxs]
                row_ind = row_ind[idxs]
                col_ind = col_ind[idxs]
        else:
            phi_width_ = highest_label if phi_dim_ is None else phi_dim_

        assert np.all(col_ind < phi_width_)
        phi_shape_ = (num_graphs, phi_width_)

        return scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=phi_shape_, dtype=phi_dtype)

    # Just count the labels in the 0-th iteration. This corresponds to the graph_labels, but indexed correctly into phi
    phi = add_labels_to_phi_(graph_labels, -1)
    phi_lists = [phi.tolil()]

    if round_signatures_to_decimals == -1:
        # 10^-1 = 0.1
        rounding_factor = 1
    else:
        rounding_factor = np.power(10, round_signatures_to_decimals)

    last_highest_label = -1
    # For the number of iterations h...
    for i in range(h):
        # ... use previous label counters/lookups, if given
        label_lookup = label_lookups[i + 1]
        label_counter = label_counters[i + 1]
        # ... go over all graphs
        for idx, (labels, adjacency_matrix) in enumerate(zip(graph_labels, adjacency_matrices)):
            # ... generate the signatures (see paper) for each graph
            signatures = np.around((labels + adjacency_matrix * log_primes[labels]), round_signatures_to_decimals).astype(np.float64)
            print(i, idx, labels, signatures)
            print(adjacency_matrix.todense())
            print()
            if ignore_label_order:
                signatures -= (labels * rounding_factor)

            # ... add missing signatures to the label lookup
            for signature in signatures:
                if signature not in label_lookup:
                    label_lookup[signature] = label_counter
                    label_counter += 1
                    # label_counter should NEVER be greater than the dimension of phi!
                    #assert truncate_to_highest_label or label_counter <= phi_shape[1]

            # ... relabel the graphs with the new (compressed) labels
            new_labels = np.array([label_lookup[signature] for signature in signatures], dtype = labels_dtype)
            graph_labels[idx] = new_labels

        phi = add_labels_to_phi_(graph_labels, i)

        non_zero = phi.nonzero()[1]

        # ... exit early when no new labels are found (= convergence)
        highest_label = np.max(phi.nonzero()[1]) if len(non_zero) else last_highest_label
        if use_early_stopping and last_highest_label == highest_label:
            break
        last_highest_label = highest_label

        # ... save phi
        phi_lists.append(phi.tolil())
        # ... save label counters/lookups for later use
        new_label_counters.append(label_counter)
        new_label_lookups.append(label_lookup)

    if use_early_stopping and fill_up_missing_iterations:
        expected_elements = (h + 1)
        diff_in_h = expected_elements - len(phi_lists)
        # When the algorithm converged...
        if diff_in_h != 0:
            # ... fill the remaining iterations
            for i in range(diff_in_h):
                phi_lists.append(phi_lists[-1])
                new_label_counters.append(new_label_counters[-1])
                new_label_lookups.append(new_label_lookups[-1])

        for x in [phi_lists, new_label_counters, new_label_lookups]:
            assert len(x) == expected_elements

    # Return the phis, the lookups and counter, so the calculation can resumed later on with new graphs
    return phi_lists, new_label_lookups, new_label_counters


def relabel_graphs(graphs: collections.abc.Iterable, label_counter: int = 0, label_lookup: dict = {}, labels_dtype: np.dtype = np.uint32, append: bool = True):
    labels = [[] for i in range(len(graphs))]
    nodes = [nodes for adjs, nodes in graphs]
    for idx, nodes_ in enumerate(nodes):
        for label in nodes_:
            if append and label not in label_lookup:
                label_lookup[label] = label_counter
                label_counter += 1
        labels[idx] = np.array([label_lookup[x] for x in nodes_], dtype=labels_dtype)
    return label_lookup, label_counter, labels
