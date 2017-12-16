import unittest
from transformers.remove_single_occurrence_graph_labels import RemoveInfrequentGraphLabels
import networkx as nx
import scipy.sparse
import numpy as np

MULTIPLE_LABEL = 'Multiple Occurrences'
MULTIPLE_LABEL_2 = 'Multiple Occurrences 2'
SINGLE_LABEL = 'Single Occurrence'

def _get_adj_matrix_for_labels(labels, set_elements = True):
    num_labels = len(labels)
    adj = scipy.sparse.lil_matrix((num_labels, num_labels), dtype=np.uint8)
    if set_elements:
        adj[:, :] = np.array(range(num_labels * num_labels)).reshape(num_labels, -1) + 1
    return adj


class RemoveSingleOccurrenceGraphLabelsTest(unittest.TestCase):
    def test_single_occurrence_remover_nxgraph(self):
        remover = RemoveInfrequentGraphLabels()

        g1 = nx.Graph()
        g1.add_node(MULTIPLE_LABEL)
        g1.add_node(MULTIPLE_LABEL_2)

        g2 = g1.copy()
        # This label is only once in the dataset
        g2.add_node(SINGLE_LABEL)

        X = [g1, g2]
        X_new = remover.transform(X)
        g1_new, g2_new = X_new

        self.assertEqual(len(X_new), len(X))
        for g in X_new:
            self.assertIsInstance(g, nx.Graph)
            self.assertTrue(g.has_node(MULTIPLE_LABEL))
            self.assertTrue(g.has_node(MULTIPLE_LABEL_2))
            self.assertEqual(g.number_of_nodes(), 2)

        self.assertFalse(g2_new.has_node(SINGLE_LABEL))
        self.assertListEqual(g1_new.nodes(), g1.nodes())
        self.assertFalse(g2_new.nodes() == g2.nodes())

    def test_single_occurrence_remover_adj_tuple(self):
        g1_labels = [MULTIPLE_LABEL, MULTIPLE_LABEL_2]
        g1_adj = _get_adj_matrix_for_labels(g1_labels)

        g2_labels = [MULTIPLE_LABEL, SINGLE_LABEL, MULTIPLE_LABEL_2]
        g2_adj = _get_adj_matrix_for_labels(g2_labels)

        X = [(g1_adj, g1_labels), (g2_adj, g2_labels)]
        remover = RemoveInfrequentGraphLabels()

        X_new = remover.transform(X)
        self.assertEqual(len(X), len(X_new))

        for x in X_new:
            self.assertIsInstance(x, tuple)
            self.assertEqual(len(x), 2)
            adj, labels = x

            num_of_multiple_labels = 2
            self.assertEqual(adj.shape, (num_of_multiple_labels, num_of_multiple_labels))
            self.assertEqual(len(labels), num_of_multiple_labels)
