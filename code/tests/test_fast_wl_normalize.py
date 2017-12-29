import unittest

from transformers import FastWLGraphKernelTransformer
from tests import helper
import numpy as np

class FastWLNormalizeTest(unittest.TestCase):

    def test_normalize_wl_without_node_weights(self):
        t = FastWLGraphKernelTransformer(
            h=2,
            use_early_stopping=False,
            truncate_to_highest_label=True,
            # This is important
            norm='l1'
        )

        X = helper.get_graphs()
        X = X[:100]

        phi_list = t.fit_transform(X)

        for phi in phi_list:
            # Assert that phi is normalized
            self.assertTrue(np.allclose(phi.sum(axis=1), np.ones(phi.shape[0])))
