import unittest
from tests import helper
from kernels import spgk
import networkx as nx

DEPTH = 10
NUM_GRAPHS = 20


class SPGKTest(unittest.TestCase):

    def test_normal(self):
        graphs = helper.get_complete_graphs(num_graphs=NUM_GRAPHS, as_tuples=False)
        gram_matrix = spgk.transform(graphs, depth=DEPTH)

        for graph_idx, row in enumerate(gram_matrix):
            for other_graph_ix, similarity in enumerate(row):
                self.assertEqual(similarity, 1)

    def test_no_similarity(self):
        graph_1 = nx.Graph()
        graph_1.add_edge('1', '2')

        graph_2 = nx.Graph()
        graph_2.add_edge('3', '4')

        graphs = [graph_1, graph_2]

        gram_matrix = spgk.transform(graphs, depth=DEPTH)

        for graph_idx, row in enumerate(gram_matrix):
            for other_graph_ix, similarity in enumerate(row):
                # If the compared graphs are the same, the score should be above 0
                if graph_idx == other_graph_ix:
                    self.assertGreater(similarity, 0)
                else:
                    self.assertEqual(similarity, 0)
