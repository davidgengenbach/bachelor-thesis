import unittest
from utils import time_utils

class TimeUtilsTest(unittest.TestCase):

    # Not neccesary to test, but...
    def test(self):
        one_minute = 60
        one_hour = one_minute * 60

        tests = [
            (one_minute,         '0:01:00'),
            (one_minute * 2,     '0:02:00'),
            (one_hour,           '1:00:00'),
            (one_hour * 1.5,     '1:30:00'),
            (one_hour * 10,     '10:00:00')
        ]

        for time_in_s, expected in tests:
            self.assertEqual(time_utils.seconds_to_human_readable(time_in_s), expected)
