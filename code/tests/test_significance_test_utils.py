import unittest
from utils import significance_test_utils
import numpy as np
import copy

class TimeUtilsTest(unittest.TestCase):

    def _get_test_result(self, metric, same_label = True):
        y_true = ['a', 'b']
        y_a = ['b', 'b']

        if same_label:
            y_b = copy.deepcopy(y_a)
        else:
            y_b = ['a', 'b']

        result = significance_test_utils.Result(y_true, [y_a, y_b])
        y_true, y_pred_a, y_pred_b = significance_test_utils.get_transformed_results(result)
        metrics = significance_test_utils.randomization_test(y_true, y_pred_a=y_pred_a, y_pred_b=y_pred_b, metric=metric, num_trails=1000)
        diffs = metrics[:, 0] - metrics[:, 1]
        return diffs


    def test_same_labels(self):
        for metric_name, metric in significance_test_utils.metrics:
            diffs = self._get_test_result(metric, same_label=True)
            # There should be no difference if the models have exactly the same predicted labels
            self.assertTrue(np.all(diffs == 0))

    def test_different_labels(self):
        for metric_name, metric in significance_test_utils.metrics:
            diffs = self._get_test_result(metric, same_label=False)
            # There should be a difference in the randomization tests when the predicted labels are not the same
            self.assertFalse(np.all(diffs == 0))