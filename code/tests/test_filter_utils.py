import unittest
import helper
from utils import filter_utils
from utils import dataset_helper

TEST_FILENAMES = [
    ('dataset_graph_cooccurrence_3_all_un-lemmatized_ling-spam.phi.npy', [
        # [(include_filter, exclude_filter, limit_dataset), expected]
        # Check limit_dataset
        [(None, None, 'ling-spam'),     True],
        [(None, None, ':)'),            False],
        # Check include_filter
        [('lemmatized', None, None),    True],
        [(':)', None, None),            False],
        # Check exclude_filter
        [(None, 'lemmatized', None),    False],
        [(None, ':)', None),            True]
    ])
]


class FilterUtilsTest(unittest.TestCase):

    def test(self):
        for filename, tests in TEST_FILENAMES:
            dataset = dataset_helper.get_dataset_name_from_graph_cachefile(filename)

            for (include_filter, exclude_filter, limit_dataset), expected in tests:
                result = filter_utils.file_should_be_processed(filename, include_filter, exclude_filter, dataset, limit_dataset)
                assert result == expected
