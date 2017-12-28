import unittest

from utils import dataset_helper, graph_helper
from transformers import FastWLGraphKernelTransformer

import numpy as np

class FastWLTruncateTest(unittest.TestCase):

    def get_graphs(self):
        if hasattr(self, 'X'):
            return self.X

        X, _ = dataset_helper.get_concept_map_for_dataset('ling-spam')
        X = graph_helper.get_graphs_only(X)
        self.X = X
        return X

    def test_normalize_wl_without_node_weights(self):
        t = FastWLGraphKernelTransformer(
            h=2,
            use_early_stopping=False,
            truncate_to_highest_label=True,
            # This is important
            norm='l1'
        )
        X = self.get_graphs()
        X = X[:100]

        phi_list = t.fit_transform(X)

        for phi in phi_list:
            # Is normalized
            self.assertTrue(np.allclose(phi.sum(axis=1), np.ones(phi.shape[0])))
