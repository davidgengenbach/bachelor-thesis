import unittest

from tests import helper
from transformers import RemoveNewGraphNodes
import networkx as nx
from utils import graph_helper
import scipy
import scipy.sparse


def get_graphs(as_adj=False):
    g_train = nx.Graph()
    g_train.add_edge('A', 'B')

    g_test = nx.Graph()
    g_test.add_edge('A', 'C')

    if as_adj:
        g_test = graph_helper.convert_graphs_to_adjs_tuples([g_test], copy=True)[0]
        g_train = graph_helper.convert_graphs_to_adjs_tuples([g_train], copy=True)[0]

    return g_train, g_test

class RemoveNewGraphNodesTest(unittest.TestCase):

    def test_remove_new_node_nx(self):
        t = RemoveNewGraphNodes()

        g_train, g_test = get_graphs()
        X_train, X_test = [g_train], [g_test]

        g_train_ = t.fit_transform(X_train)[0]
        g_test_ = t.transform(X_test)[0]

        self.assertTrue(g_train.nodes() == g_train_.nodes())
        self.assertTrue(g_test.nodes() != g_test_.nodes())

        self.assertEqual(g_test.nodes().count('C'), 1)
        # Has removed the 'C' label
        self.assertEqual(g_test_.nodes().count('C'), 0)

    def test_remove_new_node_tuple(self):
        t = RemoveNewGraphNodes()

        g_train, g_test = get_graphs(as_adj=True)
        X_train, X_test = [g_train], [g_test]

        g_train_ = t.fit_transform(X_train)[0]
        g_test_ = t.transform(X_test)[0]

        self.assertTrue(g_train[1] == g_train_[1])
        self.assertTrue(g_test[1] != g_test_[1])

        self.assertEqual(g_test[1].count('C'), 1)
        self.assertEqual(g_test_[1].count('C'), 0)

