import scipy
from scipy.sparse import lil_matrix, csr_matrix, csc_matrix, vstack
import numpy as np
import copy
import networkx as nx
import math
import gc

def add_row_and_column(mat, added = (0, 0), num = 1, dtype = None):
    mat_b = lil_matrix((mat.shape[0] + added[0], mat.shape[1] + added[1]), dtype = dtype)
    mat_b[:mat.shape[0],:mat.shape[1]] = mat.todense()
    return mat_b

def compute_phi(graph, phi_dimension, label_lookups, label_counters, h, keep_history = True, allow_new_nodes = False):
    assert len(label_lookups) == h + 1
    assert len(label_counters) == h + 1

    labels = [label_lookups[0].tolist()[node] for node in sorted(graph.nodes())]
    phi = np.zeros(phi_dimension, dtype = np.int32)
    phi_list = [0] * (h + 1) if keep_history else [0]
    phi_list[0] = phi
    nodes = graph.nodes()

    # If the graph is empty, just return the empty phi and the initial lookups/counters
    if not len(nodes):
        return phi_list, label_lookups, label_counters

    adj_mat = nx.adjacency_matrix(graph, nodelist = sorted(graph.nodes()))
    for label in labels:
        phi[label] += 1

    new_label_lookups = [label_lookups[0].tolist()]
    new_label_counters = [label_counters[0]]
    new_labels = np.copy(labels)
    for it in range(h):
        long_labels = np.copy(labels)
        label_lookup = label_lookups[it + 1]
        if not isinstance(label_lookup, dict):
            label_lookups = label_counter.tolist()
        label_counter = label_counters[it + 1]
        phi = np.zeros(phi_dimension, dtype = np.int32)
        num_nodes = len(graph.nodes())
        for node_idx in range(num_nodes):
            long_label = [labels[node_idx]]
            neighbors = np.argwhere(adj_mat[node_idx] > 0)
            if len(neighbors):
                long_label += sorted([long_labels[node[1]] for node in neighbors])
            long_label = tuple(long_label)
            if not long_label in label_lookup:
                label_lookup[long_label] = str(label_counter)
                new_labels[node_idx] = str(label_counter)
                label_counter += 1
            else:
                new_labels[node_idx] = label_lookup[long_label]
        new_label_lookups.append(label_lookup)
        new_label_counters.append(label_counter)
        aux = np.bincount(new_labels)
        phi[new_labels] += aux[new_labels]
        phi_list[it + 1 if keep_history else 0] = phi
        labels = np.copy(new_labels)
    return phi_list, new_label_lookups, new_label_counters


def concatenate_csc_matrices_by_columns(matrix1, matrix2):
    new_data = np.concatenate((matrix1.data, matrix2.data))
    new_indices = np.concatenate((matrix1.indices, matrix2.indices))
    new_ind_ptr = matrix2.indptr + len(matrix1.data)
    new_ind_ptr = new_ind_ptr[1:]
    new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

    return csc_matrix((new_data, new_indices, new_ind_ptr))


def WL_compute_batched(adjs, node_label, all_nodes, h, initial_label_counters = None, initial_label_lookups = None, phi_dim = None, batch_size = 1500, keep_phi_history = True, gc_after_each = True, **wl_params):
    """Computes the Weisfeiler-Lehman kernel for the given graphs in batches.
    
    Args:
        adjs (list(sparse_matrix)): the adjacency matrices
        node_label (list(str)): the node labels of the graphs
        all_nodes (list(str)): all node labels
        batch_size (int): number of graphs per batch
        h (int): number of iterations of WL
        **wl_params: all other parameter for WL_compute
    
    Returns:
        tuple(phi_list, current_label_lookups, current_label_counters): as follows:
            - phi_list: contains for each iteration the phis
            - current_label_lookups: list(dict) of label lookups
            - current_label_counters: list(int) the counters
    """
    assert len(adjs) == len(node_label)
    num_elements = len(adjs)
    num_batches = math.ceil(num_elements / batch_size)

    if not phi_dim:
        phi_dim = sum(len(x) for x in node_label) + len(all_nodes)

    current_label_counters = initial_label_counters
    current_label_lookups = initial_label_lookups

    phi_lists = []
    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, num_elements)
        K, phi_list, current_label_lookups, current_label_counters = WL_compute(adjs[start:end], node_label[start:end], initial_label_counters= current_label_counters, initial_label_lookups= current_label_lookups, all_nodes=all_nodes, h = h, phi_dim = phi_dim, keep_phi_history = keep_phi_history, **wl_params)
        phi_dim = phi_list[-1].shape[0]
        phi_lists.append(phi_list)
        if gc_after_each:
            gc.collect()
    phi_shape = (phi_lists[0][0].shape[0], num_elements)
    # phi_lists is an list of lists of lil_matrices
    phi_list = [None for i in range(h + 1)]
    for phi_list_ in phi_lists:
        for it, phi in enumerate(phi_list_):
            if phi_list[it] is None:
                phi_list[it] = phi.tocsc()
            else:
                phi_list[it] = scipy.sparse.hstack([phi_list[it], phi])
                # This is nice, but has to be converted to csc matrices
                #phi_list[it] = concatenate_csc_matrices_by_columns(phi_list[it].tocsc(), phi.tocsc())
            # Slowest
            #phi_list[it][:, offset:offset + num_graphs_in_list] = phi
    if gc_after_each:
        gc.collect()
    return phi_list, current_label_lookups, current_label_counters

def WL_compute(ad_list, node_label, h, all_nodes = (), compute_k = True, initial_label_lookups = None, initial_label_counters = None, phi_dim = None, keep_phi_history = True, DEBUG = False):
    """Computes weisfeiler lehman kernel for given graphs.
    
    Args:
        ad_list (list(matrix)): The adjadency matrices of the graphs
        node_label (list(str)): The node labels for the graphs
        h (int): iterations
        all_nodes (tuple, optional): all node label names
        DEBUG (bool, optional): Add verbosity
    
    Returns:
        list(matrix): K (Kernel matrices)
        list(matrix): phi_list (each column is the feature vector for a graph)
    """
    # Total number of graphs in the dataset
    n = len(ad_list)

    assert isinstance(ad_list, list)
    assert isinstance(node_label, list)
    assert isinstance(h, int)
    assert len(ad_list) == len(node_label)
    
    # Total number of nodes in dataset: initialized as zero
    tot_nodes = sum(len(x) for x in node_label) + len(all_nodes)
    # list of kernel matrices
    K = [0] * (h+1)
    # list of feature mtrices
    phi_list = [0] * (h+1) if keep_phi_history else [0]

    if not phi_dim:
        phi_dim = tot_nodes

    phi_shape = (phi_dim, n)
    
    #each column of phi will be the explicit feature representation for the graph j
    phi = lil_matrix(phi_shape, dtype = np.int32)

    # labels will be used to store the new labels
    labels = [0] * n

    #label lookup is a dictionary which will contain the mapping
    # from multiset labels (strings) to short labels (integers)
    label_lookup = initial_label_lookups[0] if initial_label_lookups else {}
    if not isinstance(label_lookup, dict):
        label_lookup = label_lookup.tolist()
    # Save the counters and lookups for later iterations
    label_counters = []
    label_lookups = []

    # counter to create possibly new labels in the update step
    label_counter = initial_label_counters[0] if initial_label_counters else 0
    for label in all_nodes:
        if label not in label_lookup:
            label_lookup[label] = label_counter
            label_counter += 1
        
    # Note: here we are just renaming the node labels from 0,..,num_labels
    # for each graph
    for i in range(n):
        # copy the original labels
        l_aux = np.copy(node_label[i])

        # will be used to store the new labels
        labels[i] = np.zeros(len(l_aux), dtype = np.int32)

        # for each label in graph
        for j in range(len(l_aux)):
            l_aux_str = str(l_aux[j])

            # If the string do not already exist
            # then create a new short label
            if not l_aux_str in label_lookup:
                label_lookup[l_aux_str] = label_counter
                labels[i][j] = label_counter
                label_counter += 1
            else:
                labels[i][j] = label_lookup[l_aux_str]
            # node histograph of the new labels
            phi[labels[i][j],i] += 1

    label_counters.append(np.copy(label_counter))
    label_lookups.append(np.copy(label_lookup))

    L = label_counter
    if DEBUG: print('Number of original labels %d' %L)
    #####################
    # --- Main code --- #
    #####################

    # Now we are starting with the first iteration of WL

    # features obtained from the original node (renamed) labels
    phi_list[0] = phi

    # Kernel matrix based on original features
    if compute_k: K[0] = phi.transpose().dot(phi).toarray().astype(np.float32)
    
    if DEBUG: print("K original is computed")
    
    # Initialize iterations to 0
    it = 0

    # copy of the original labels: will stored the new labels
    new_labels = np.copy(labels)
    # until the number of iterations is less than h
    while it < h:
        # Initialize dictionary and counter 
        # (same meaning as before)        
        #label_lookup = {}
        #label_counter = 0

        label_lookup = initial_label_lookups[it + 1] if initial_label_lookups else {}
        if not isinstance(label_lookup, dict):
            label_lookups = label_counter.tolist()
        label_counter = initial_label_counters[it + 1] if initial_label_counters else 0

        # Initialize phi as a sparse matrix
        phi = lil_matrix(phi_shape, dtype = np.int32)

        if DEBUG: print("Iteration %d: phi is computed" % it)

        # for each graph in the dataset
        for i in range(n):
            if DEBUG and i % int(n / 2) == 0: print('\tGraph {:>10}/{}'.format(i, n))
            # will store the multilabel string
            l_aux_long = np.copy(labels[i])

            # for each node in graph
            for v in range(ad_list[i].shape[0]):
                # the new labels convert to tuple
                long_label = [l_aux_long[v]]
                neighbors = np.argwhere(ad_list[i][v] > 0).tolist()
                if len(neighbors):
                    long_label += sorted([l_aux_long[node[0]] for node in neighbors])
                long_label = tuple(long_label)
                if not long_label in label_lookup:
                    label_lookup[long_label] = str(label_counter)
                    new_labels[i][v] = str(label_counter)
                    label_counter += 1
                else:
                    new_labels[i][v] = label_lookup[long_label]
            # count the node label frequencies
            aux = np.bincount(new_labels[i])
            phi[new_labels[i],i] += aux[new_labels[i]].reshape(-1, 1)
        phi_idx = it+1 if keep_phi_history else 0
        # create phi for iteration it+1
        phi_list[phi_idx] = phi

        # create K at iteration it+1
        if compute_k: K[phi_idx] = K[it] + phi.transpose().dot(phi).toarray().astype(np.float32)
        
        # Initialize labels for the next iteration as the new just computed
        labels = copy.deepcopy(new_labels)

        # increment the iteration
        it += 1 

        label_lookups.append(label_lookup)
        label_counters.append(label_counter)
    return K, phi_list, label_lookups, label_counters
