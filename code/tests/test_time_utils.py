import unittest
from utils import time_utils

class TimeUtilsTest(unittest.TestCase):

    # Not neccesary to test, but...
    def test(self):
        self.assertEqual(time_utils.seconds_to_human_readable(60), '0:01:00')
        self.assertEqual(time_utils.seconds_to_human_readable(120), '0:02:00')
        self.assertEqual(time_utils.seconds_to_human_readable(3600), '1:00:00')