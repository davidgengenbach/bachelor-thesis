import unittest
import collections
from utils import filename_utils

TestItem = collections.namedtuple('TestItem', ['filename', 'expected_dataset', 'ignore_subtype'])


FILENAMES = [
    TestItem('data/results/dataset_graph_cooccurrence_1_all_un-lemmatized_ling-spam.simple.gram.npy', 'ling-spam',
             False),
    TestItem('data/results/dataset_graph_cooccurrence_1_all_un-lemmatized_ling-spam.spgk-1.gram.npy', 'ling-spam',
             False),
    TestItem('data/results/dataset_graph_cooccurrence_1_all_un-lemmatized_ng20.simple.gram.npy', 'ng20', False),
    TestItem('data/results/dataset_graph_cooccurrence_1_all_un-lemmatized_ng20-ana.spgk-1.gram.npy', 'ng20-ana', False),
    TestItem('dataset_graph_cooccurrence_1_all_un-lemmatized_webkb.spgk-1.gram.npy.results.npy', 'webkb', False),
    TestItem('dataset_graph_gml_ng20-single.npy', 'ng20', False),
    TestItem('dataset_graph_gml_ng20-ana.npy', 'ng20-ana', False),
    TestItem('dataset_graph_gml_ng20-ana.npy', 'ng20', True),
    TestItem('not a dataset', None, False),
]

class FilenameUtilsTst(unittest.TestCase):
    def test(self):
        for test_item in FILENAMES:
            self.assertEqual(filename_utils.get_dataset_from_filename(test_item.filename, test_item.ignore_subtype), test_item.expected_dataset)