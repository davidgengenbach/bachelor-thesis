import unittest

from transformers import FastWLGraphKernelTransformer

from tests import helper
from utils import graph_helper

import networkx as nx

def get_graphs(as_adj=False):
    g_train = nx.DiGraph()
    g_train.add_edge('A', 'B')
    g_train.add_edge('A', 'C')

    g_test = nx.Graph()
    g_train.add_edge('A', 'B')
    g_train.add_edge('A', 'C')
    g_train.add_edge('B', 'C')

    if as_adj:
        g_test = graph_helper.convert_graphs_to_adjs_tuples([g_test], copy=True)[0]
        g_train = graph_helper.convert_graphs_to_adjs_tuples([g_train], copy=True)[0]

    return g_train, g_test


class FastWLMatchesTest(unittest.TestCase):

    def get_phi_list(self, **fast_wl_params):
        t = FastWLGraphKernelTransformer(**fast_wl_params)
        X = helper.get_graphs()

        phi_list = t.fit_transform(X)
        return phi_list

    def test_something(self):
        phis = self.get_phi_list(h=1)
        print(phis)