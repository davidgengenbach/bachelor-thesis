"""
Created on Fri Apr 14 2017

@author: g.nikolentzos

"""

import sys
import numpy as np
import networkx as nx


def spgk(sp_g1, sp_g2, norm1, norm2):
    if norm1 == 0 or norm2 == 0:
        return 0
    else:
        kernel_value = 0
        for node1 in sp_g1:
            if node1 in sp_g2:
                kernel_value += 1
                for node2 in sp_g1[node1]:
                    if node2 != node1 and node2 in sp_g2[node1]:
                        kernel_value += (1.0/sp_g1[node1][node2]) * (1.0/sp_g2[node1][node2])

        kernel_value /= (norm1 * norm2)

        return kernel_value


def build_kernel_matrix(graphs, depth):
    sp = []
    norm = []

    for g in graphs:
        current_sp = nx.all_pairs_dijkstra_path_length(g, cutoff=depth, weight = 'NO_WEIGHT_USED')
        sp.append(current_sp)

        sp_g = nx.Graph()
        for node in current_sp:
            for neighbor in current_sp[node]:
                if node == neighbor:
                    sp_g.add_edge(node, node, weight=1.0)
                else:
                    sp_g.add_edge(node, neighbor, weight=1.0/current_sp[node][neighbor])

        M = nx.to_numpy_matrix(sp_g)
        norm.append(np.linalg.norm(M, 'fro'))

    K = np.zeros((len(graphs), len(graphs)))

    for i in range(len(graphs)):
        for j in range(i, len(graphs)):
            K[i, j] = spgk(sp[i], sp[j], norm[i], norm[j])
            K[j, i] = K[i, j]
    return K
