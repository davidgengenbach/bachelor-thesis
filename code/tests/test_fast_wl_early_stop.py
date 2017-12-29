import unittest

from tests import helper
from transformers import FastWLGraphKernelTransformer


class FastWLEarlyStopTest(unittest.TestCase):

    def test_early_stop(self):
        t = FastWLGraphKernelTransformer(
            h=10,
            use_early_stopping=True,
            truncate_to_highest_label=True
        )

        X = helper.get_graphs()
        X = X[:100]
        phi_list = t.fit_transform(X)

        feature_nums = [phi.shape[1] for phi in phi_list]
        self.assertEqual(len(set(feature_nums)), len(feature_nums))
