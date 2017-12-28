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

    @unittest.skip('')
    def test_truncated_no_phi_dim(self):
        t = FastWLGraphKernelTransformer(h=2, use_early_stopping=False, truncate_to_highest_label=True)
        X = self.get_graphs()

        phi_list = t.fit_transform(X)

        for phi in phi_list:
            self.assertGreater(phi.sum(axis=1)[-1], 0)


    def test_not_truncated_with_phi_dim(self):
        phi_dim = 999999
        t = FastWLGraphKernelTransformer(h=2, use_early_stopping=False, truncate_to_highest_label=False, phi_dim=phi_dim)
        X = self.get_graphs()

        phi_list = t.fit_transform(X)

        self.assertTrue(np.all([phi.shape[1] == phi_dim for phi in phi_list]))

    def test_truncated_with_phi_dim(self):
        phi_dim = 999999
        t = FastWLGraphKernelTransformer(h=2, use_early_stopping=False, truncate_to_highest_label=True, phi_dim=phi_dim)
        X = self.get_graphs()

        phi_list = t.fit_transform(X)
        self.assertTrue(np.all([phi.shape[1] == phi_dim for phi in phi_list]))

    def test_truncated_with_small_phi_dim(self):
        phi_dim = 1000
        t = FastWLGraphKernelTransformer(h=2, use_early_stopping=False, truncate_to_highest_label=True, phi_dim=phi_dim)
        X = self.get_graphs()

        phi_list = t.fit_transform(X)
        self.assertTrue(np.all([phi.shape[1] == phi_dim for phi in phi_list]))

    def test_truncated_with_no_phi_dim_with_transform(self):
        t = FastWLGraphKernelTransformer(h=2, use_early_stopping=False, truncate_to_highest_label=True)
        X = self.get_graphs()

        num_train = 1000
        X_train, X_test = X[:num_train], X[num_train:]

        phi_list = t.fit_transform(X_train)
        phi_list_test = t.transform(X_test)
        for phi_train, phi_test in zip(phi_list, phi_list_test):
            self.assertEqual(phi_train.shape[1], phi_test.shape[1])
