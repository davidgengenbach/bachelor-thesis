import networkx as nx
import sympy
import numpy as np
from scipy.sparse import lil_matrix, coo_matrix
from collections import defaultdict
import primes


primes_arguments_required = [2, 3, 7, 19, 53, 131, 311, 719, 1619, 3671, 8161, 17863, 38873, 84017, 180503, 386093, 821641, 1742537, 3681131, 7754077, 16290047]


def fast_wl_compute(graphs, h=1, label_lookups=None, label_counters=None, primes_arguments_required=primes_arguments_required):
    new_label_lookups = []
    new_label_counters = []
    if not label_lookups:
        label_lookups = [{} for x in range(h + 1)]
    if not label_counters:
        label_counters = [1] * (h + 1)

    print('Relabeling graphs')
    label_lookup, label_counter, relabeled_labels, adjs = relabel_graphs(graphs, label_lookups[0])
    print('Relabeling graphs: END')

    assert len(relabeled_labels) == len(graphs)

    new_label_counters.append(label_counter)
    new_label_lookups.append(label_lookup)

    num_labels = len(label_lookup.keys())
    num_graphs = len(graphs)

    primes_needed = primes_arguments_required[int(np.ceil(np.log2(num_labels)) + 2)]

    log_primes = primes.get_log_primes(1, primes_needed)

    #labels = [sorted(g.nodes()) for g in graphs]
    #adjs = [nx.adjacency_matrix(graph, nodelist=labels_) for graph, labels_ in zip(graphs, relabeled_labels)]
    phi_shape = (sum(len(x) for x in relabeled_labels), num_graphs)
    # Uncomment when gram/kernel matrix should be calculated
    #K = np.zeros((num_graphs, num_graphs), dtype = np.uint32)
    phi_lists = []
    print('Starting iterations')
    for i in range(h):
        label_lookup = label_lookups[i + 1]
        label_counter = label_counters[i + 1]
        phi = lil_matrix(phi_shape, dtype=np.uint8)
        for idx, (labels_, A) in enumerate(zip(relabeled_labels, adjs)):
            signatures = np.round((labels_ + A * log_primes[np.array(labels_) - 1]), decimals=10)
            new_labels = np.zeros(len(signatures), dtype=np.uint8)
            for idx_, signature in enumerate(signatures):
                if signature not in label_lookup:
                    label_lookup[signature] = label_counter
                    label_counter += 1
                new_labels[idx_] = label_lookup[signature]
            relabeled_labels[idx] = new_labels
            phi[new_labels - 1, idx] = np.ones(phi[new_labels - 1, idx].shape, dtype=np.uint8)
        phi_lists.append(phi)
        new_label_counters.append(label_counter)
        new_label_lookups.append(label_lookups)
        # Uncomment when gram/kernel matrix is wanted
        #K += phi.T.dot(phi)
    return phi_lists, new_label_lookups, new_label_counters


def relabel_graphs(graphs, label_lookup={}):
    label_counter = 1
    labels = [[] for i in range(len(graphs))]
    adjs = [[] for i in range(len(graphs))]
    for idx, graph in enumerate(graphs):
        nodes = list(graph.nodes())
        for label in nodes:
            if label not in label_lookup:
                label_lookup[label] = label_counter
                label_counter += 1
        labels[idx] = [label_lookup[x] for x in nodes]
        adjs[idx] = nx.adjacency_matrix(graph, nodelist = nodes)
        #nx.relabel_nodes(graph, {k: v for k, v in label_lookup.items() if k in graph}, copy=False)
    return label_lookup, label_counter, labels, adjs
