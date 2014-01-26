import unittest
from analysis import filter_repeating_ticks


class TestAnalysisFunctions(unittest.TestCase):

    def test_filter_repeating_timestamps(self):
        ticks = [{'time': 1, 'last': 1}, {'time': 1, 'last': 1}]

        self.assertEqual([{'time': 1, 'last': 1}], filter_repeating_ticks(ticks))

    def test_filter_repeating_timestamps_none(self):
        self.assertEquals([], filter_repeating_ticks([]))

    def test_filter_repeating_timestamps_2(self):
        ticks = [{'time': 1, 'last': 1}, {'time': 1, 'last': 1}, {'time': 1, 'last': 2}]

        self.assertEqual([{'time': 1, 'last': 1}, {'time': 1, 'last': 2}], filter_repeating_ticks(ticks))

    def test_filter_repeating_timestamps_3(self):
        ticks = [{'time': 1, 'last': 1}, {'time': 1, 'last': 1}, {'time': 1, 'last': 2},
                 {'time': 2, 'last': 2}, {'time': 2, 'last': 2}]

        self.assertEqual([{'time': 1, 'last': 1}, {'time': 2, 'last': 2}], filter_repeating_ticks(ticks))

    def test_filter_repeating_timestamps_4(self):
        ticks = [{'time': 1, 'last': 1}, {'time': 1, 'last': 1}, {'time': 1, 'last': 2},
                 {'time': 2, 'last': 2}, {'time': 2, 'last': 2}, {'time': 2, 'last': 3}, {'time': 2, 'last': 3}]

        self.assertEqual([{'time': 1, 'last': 1}, {'time': 2, 'last': 2}, {'time': 2, 'last': 3}],
                         filter_repeating_ticks(ticks))