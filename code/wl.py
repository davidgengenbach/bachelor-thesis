from scipy.sparse import lil_matrix, csr_matrix, vstack
import numpy as np
import copy
import psutil
import networkx as nx

def add_row_and_column(mat, added = (0, 0), num = 1, dtype = None):
    mat_b = lil_matrix((mat.shape[0] + added[0], mat.shape[1] + added[1]), dtype = dtype)
    mat_b[:mat.shape[0],:mat.shape[1]] = mat.todense()
    return mat_b

def compute_phi(graph, phi_shape, label_lookups, label_counters, h):
    num_nodes = len(graph.nodes())
    labels = [label_lookups[0].tolist()[node] for node in sorted(graph.nodes())]
    phi = np.zeros(phi_shape[0], dtype = np.int32)
    phi_list = [0] * (h + 1)
    adj_mat = nx.adjacency_matrix(graph, nodelist = sorted(graph.nodes()))
    for label in labels:
        phi[label] += 1
    phi_list[0] = phi
    new_labels = np.copy(labels)
    for it in range(h):
        long_labels = np.copy(labels)
        label_lookup = label_lookups[it + 1].tolist()
        label_counter = label_counters[it + 1]
        phi = np.zeros(phi_shape[0], dtype = np.int32)
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
        aux = np.bincount(new_labels)
        phi[new_labels] += aux[new_labels]
        phi_list[it + 1] = phi
        labels = np.copy(new_labels)
    return phi_list


def WL_compute(ad_list, node_label, h, all_nodes = (), compute_k = True, DEBUG = False):
    """Computes weisfeiler lehman of the given graphs
    
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
    # CHANGE
    label_lookups = []
    # Total number of graphs in the dataset
    n = len(ad_list)
    
    # Total number of nodes in dataset: initialized as zero
    tot_nodes = len(all_nodes) + sum(len(x) for x in node_label)
    # list of kernel matrices
    K = [0]*(h+1)
    # list of feature mtrices
    phi_list = [0] * (h+1)
    
    #each column of phi will be the explicit feature representation for the graph j
    phi = lil_matrix((tot_nodes, n), dtype = np.int32)

    # labels will be used to store the new labels
    labels = [0] * n

    #label lookup is a dictionary which will contain the mapping
    # from multiset labels (strings) to short labels (integers)
    label_lookup = {}

    label_counters = []

    # counter to create possibly new labels in the update step
    label_counter = 0
    
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
        label_lookup = {}
        label_counter = 0

        # Initialize phi as a sparse matrix
        phi = lil_matrix((tot_nodes, n), dtype = np.int32)
        # convert it to array
        #phi = phi.toarray()

        if DEBUG: print("Iteration %d: phi is computed" % it)

        # for each graph in the dataset
        for i in range(n):
            if DEBUG and i % 1000 == 0: print('\tGraph {:>10}/{}'.format(i, n))
            # will store the multilabel string
            l_aux_long = np.copy(labels[i])

            # for each node in graph
            for v in range(ad_list[i].shape[0]):

                # the new labels convert to tuple
                long_label = [l_aux_long[v]]

                neighbors = np.argwhere(ad_list[i][v] > 0).tolist()
                if len(neighbors):
                    long_label += sorted([l_aux_long[node[1]] for node in neighbors])
                long_label = tuple(long_label)
                if not long_label in label_lookup:
                    label_lookup[long_label] = str(label_counter)
                    new_labels[i][v] = str(label_counter)
                    label_counter += 1
                else:
                    new_labels[i][v] = label_lookup[long_label]
                '''
                    # form a multiset label of the node neighbors 
                    #new_ad = np.zeros(len(ad_list[i][v]))
                    new_ad = np.argwhere(ad_list[i][v] > 0).tolist()
                    if len(new_ad):
                        # long labels: original node plus sorted neughbors
                        ad_aux = [l_aux_long[j[0]] for j in new_ad]
                        for label in sorted(ad_aux):
                            long_label.append(label)
                    long_label = tuple(long_label)
                    # if the multiset label has not yet occurred , add
                    # it to the lookup table and assign a number to it
                    if not long_label in label_lookup:
                        label_lookup[long_label] = str(label_counter)
                        new_labels[i][v] = str(label_counter)
                        label_counter += 1
                    # else assign it the already existing number
                    else:
                        new_labels[i][v] = label_lookup[long_label]
                    '''
            # count the node label frequencies
            aux = np.bincount(new_labels[i])
            phi[new_labels[i],i] += aux[new_labels[i]].reshape(-1, 1)
        
        L = label_counter
        if DEBUG: print('Number of compressed labels %d' % L)

        # create phi for iteration it+1
        phi_list[it+1] = phi

        if DEBUG: print("Itaration %d: phi computed" % it)
        # create K at iteration it+1
        if compute_k: K[it+1] = K[it] + phi.transpose().dot(phi).toarray().astype(np.float32)
        
        # Initialize labels for the next iteration as the new just computed
        labels = copy.deepcopy(new_labels)

        # increment the iteration
        it = it + 1 

        label_lookups.append(label_lookup)
        label_counters.append(label_counter)
    return K, phi_list, label_lookups, label_counters



def WL_compute_new(ad_list, node_label, h, k_prev, phi_prev, label_lookups_prev, label_counters_prev, all_nodes = (), DEBUG = False):
    current_index = 0

    num_graphs = len(ad_list)
    tot_nodes = len(all_nodes) + sum(len(x) for x in node_label)

    label_counter = label_counters_prev[0]

    num_graph_orig = phi_prev[0].shape[1]
    phi_added = (sum(len(x) for x in node_label), num_graphs)
    phi = add_row_and_column(phi_prev[current_index], added = phi_added, dtype = np.int32)

    current_label_lookup = label_lookups_prev[current_index].tolist()
    labels = [0] * num_graphs
    for i, node_labels in enumerate(node_label):
        labels[i] = np.zeros(len(node_labels), dtype = np.int32)
        for j, label in enumerate(node_labels):
            assert label in current_label_lookup.keys()
            label_id = current_label_lookup[label]
            labels[i][j] = label_id
            phi[labels[i][j], i + num_graph_orig ] += 1
    phi_list = [0] * (h + 1)
    phi_list[0] = phi
    K = [0] * (h + 1)
    K[0] = phi.transpose().dot(phi).astype(np.float32)
    new_labels = np.copy(labels)

    for it in range(h):
        current_index += 1
        label_lookup = label_lookups_prev[current_index].tolist()
        label_counter = label_counters_prev[current_index]
        phi = phi_prev[current_index]
        phi = add_row_and_column(lil_matrix(phi), phi_added, dtype = np.int32)
        for graph_idx in range(num_graphs):
            l_aux_long = np.copy(labels[graph_idx])
            for v in range(len(ad_list[graph_idx])):
                new_node_label = tuple([l_aux_long[v]]) 
                num_nodes = len(ad_list[graph_idx][v])
                new_ad = np.argwhere(ad_list[i][v] > 0)
                if len(new_ad):
                    ad_aux = tuple([l_aux_long[int(j)] for j in new_ad])
                    long_label = tuple(tuple(new_node_label)+tuple(sorted(ad_aux)))
                else:
                    long_label = tuple(new_node_label)
                # if the multiset label has not yet occurred , add
                # it to the lookup table and assign a number to it
                if not long_label in label_lookup:
                    label_lookup[long_label] = str(label_counter)
                    new_labels[graph_idx][v] = str(label_counter)
                    label_counter += 1
                else:
                    new_labels[graph_idx][v] = label_lookup[long_label]
            aux = np.bincount(new_labels[graph_idx])
            phi[new_labels[graph_idx], num_graph_orig + graph_idx] += aux[new_labels[graph_idx]].reshape(-1, 1)
        phi_sparse = lil_matrix(phi)
        phi_list[it + 1] = phi_sparse
        K[it + 1] = K[it] + phi_sparse.transpose().dot(phi_sparse).astype(np.float32)
        #assert np.array_equal(K[it + 1][:num_graph_orig,:num_graph_orig], k_prev[it + 1])
        labels = copy.deepcopy(new_labels)
    return K, phi_list