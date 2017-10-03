import unittest
from tests import helper
from kernels import fast_wl

H = 10
NUM_GRAPHS = 20


class FastWLTest(unittest.TestCase):

    def test_normal(self):
        graphs = helper.get_complete_graphs(num_graphs=NUM_GRAPHS, as_tuples=True)

        self.calculate_fast_wl(graphs)

        for iteration, (phi, lookup, counter) in self.iterate_over_wl_iterations():
            for (adj, labels), row in zip(self.graphs, phi):
                self.assertEqual(len(row.nonzero()[0]), len(labels))

    def test_same_label_per_graph(self):
        graphs = helper.get_complete_graphs(num_graphs=NUM_GRAPHS, as_tuples=True)

        # Give all graphs the same labels
        for adj, labels in graphs:
            for idx, label in enumerate(labels):
                labels[idx] = 'label_{}'.format(idx)

        self.calculate_fast_wl(graphs)
        self.check_all_phi_rows_same()

        for iteration, (phi, lookup, counter) in self.iterate_over_wl_iterations():
            # All labels should be the same for all iterations
            self.assertEqual(len(lookup.keys()), len(self.get_all_graph_labels()))

    def test_same_label(self):
        graphs = helper.get_complete_graphs(num_graphs=NUM_GRAPHS, as_tuples=True)

        # Give all graphs the same labels
        for adj, labels in graphs:
            for idx, label in enumerate(labels):
                labels[idx] = 'only_one_label'

        self.calculate_fast_wl(graphs)
        self.check_all_phi_rows_same()

        for iteration, (phi, lookup, counter) in self.iterate_over_wl_iterations():
            # All labels should be the same for all iterations
            self.assertEqual(len(lookup.keys()), 1)


    def check_all_phi_rows_same(self):
        for iteration, (phi, lookup, counter) in self.iterate_over_wl_iterations():
            for graph_idx, row in enumerate(phi[:-1]):
                next_row = phi[graph_idx + 1]
                self.assertListEqual(row.toarray()[0].tolist(), next_row.toarray()[0].tolist())

    def get_all_graph_labels(self):
        labels = set()
        for _, labels_ in self.graphs:
            labels |= set(labels_)
        return labels

    def calculate_fast_wl(self, graphs, check_results=True):
        self.graphs = graphs
        self.phi_lists, self.label_lookups, self.label_counters = fast_wl.transform(graphs, h=H)

        if check_results:
            self.check_wl_returns()

    def iterate_over_wl_iterations(self):
        for iteration, (phi, lookup, counter) in enumerate(
                zip(self.phi_lists, self.label_lookups, self.label_counters)):
            phi = phi.T
            yield iteration, (phi, lookup, counter)

    def check_wl_returns(self):
        self.assertEqual(len(self.phi_lists), H + 1)
        self.assertEqual(len(self.label_lookups), len(self.label_counters))

        for iteration, (phi, lookup, counter) in self.iterate_over_wl_iterations():
            self.assertEqual(len(lookup.keys()), counter)
            self.assertEqual(phi.shape[0], len(self.graphs))
