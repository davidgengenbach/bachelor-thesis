import unittest

from utils import graph_helper
import os
from kernels import fast_wl
import sklearn
from transformers.phi_picker_transformer import PhiPickerTransformer
from transformers.gram_matrix_transformer import GramMatrixTransformer
from sklearn import model_selection
from sklearn import linear_model
from sklearn import pipeline
from sklearn import svm

from transformers.fast_wl_graph_kernel_transformer import FastWLGraphKernelTransformer

CURRENT_DIR = os.path.abspath(__file__).rsplit('/', 1)[0]

ENZYME_DIR = '{}/data/enzymes'.format(CURRENT_DIR)

H = 2
class GraphHelperTestCase(unittest.TestCase):

    def test(self):
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