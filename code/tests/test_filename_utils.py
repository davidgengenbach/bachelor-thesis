import unittest
from utils import filename_utils


FILENAMES = [
    ['data/results/dataset_graph_cooccurrence_1_all_un-lemmatized_ling-spam.simple.gram.npy', 'ling-spam'],
    ['data/results/dataset_graph_cooccurrence_1_all_un-lemmatized_ling-spam.spgk-1.gram.npy', 'ling-spam'],
    ['data/results/dataset_graph_cooccurrence_1_all_un-lemmatized_ng20.simple.gram.npy', 'ng20'],
    ['data/results/dataset_graph_cooccurrence_1_all_un-lemmatized_ng20-ana.spgk-1.gram.npy', 'ng20-ana'],
    ['dataset_graph_cooccurrence_1_all_un-lemmatized_webkb.spgk-1.gram.npy.results.npy', 'webkb'],
    ['not a dataset', None]
]

class FilenameUtilsTst(unittest.TestCase):
    def test(self):
        for filename, expected_dataset in FILENAMES:
            self.assertEqual(filename_utils.get_dataset_from_filename(filename), expected_dataset)