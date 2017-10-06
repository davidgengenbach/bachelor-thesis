import unittest
import os

import sklearn
from transformers.phi_picker_transformer import PhiPickerTransformer
from transformers.gram_matrix_transformer import GramMatrixTransformer
from transformers.fast_wl_graph_kernel_transformer import FastWLGraphKernelTransformer
from sklearn import model_selection
from sklearn import linear_model
from sklearn import pipeline
from sklearn import svm
from kernels import fast_wl

from utils import graph_helper, dataset_helper, filename_utils

CURRENT_DIR = os.path.abspath(__file__).rsplit('/', 1)[0]

ENZYME_DIR = '{}/data/enzymes'.format(CURRENT_DIR)

H = 2
class GraphHelperTestCase(unittest.TestCase):

    def test_combined_concept_graph_texts(self):
        for graph_dataset in dataset_helper.get_all_cached_graph_datasets():
            if '_v2' not in graph_dataset: continue
            dataset_name = filename_utils.get_dataset_from_filename(graph_dataset)
            with self.subTest(graph_dataset = graph_dataset, dataset_name = dataset_name):
                X_combined, Y_combined = graph_helper.get_filtered_text_graph_dataset(graph_dataset)
                for (graph, text, y_id), y in zip(X_combined, Y_combined):
                    for node, data in graph.nodes(data=True):
                        self.assertEqual(node, data['name'])


    @unittest.skip("Not used")
    def test_mutag_enzyme_graphs(self):
        X, Y = graph_helper.get_graphs_with_mutag_enzyme_format(ENZYME_DIR)

        num_vertices = sum([len(labels) for _, labels in X])

        estimator = sklearn.pipeline.Pipeline([
            ('fast_wl', FastWLGraphKernelTransformer(h=H, should_cast=False, remove_missing_labels = True, phi_dim = num_vertices)),
            ('phi_picker', PhiPickerTransformer(return_iteration='stacked')),
            #('gram_matrix', GramMatrixTransformer()),
            ('clf', sklearn.svm.LinearSVC(class_weight = 'balanced'))
        ])

        scores = sklearn.model_selection.cross_val_score(estimator, X, Y, cv = 3, scoring = 'accuracy')
        print(scores)