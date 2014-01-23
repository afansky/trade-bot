import unittest
from analysis import filter_repeating_ticks


class TestAnalysisFunctions(unittest.TestCase):

    def test_filter_repeating_timestamps(self):
        ticks = [{'time': 1, 'last': 1}, {'time': 1, 'last': 1}]

        self.assertEqual([{'time': 1, 'last': 1}], filter_repeating_ticks(ticks))