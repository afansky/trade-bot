import unittest
from simulator import merge_timestamps


class TestSimulatorFunctions(unittest.TestCase):

    def merge_timestamps_test(self):
        btc_usd = [{'time': 1, 'last': 1}, {'time': 3, 'last': 2}]
        ltc_usd = [{'time': 1, 'last': 25}, {'time': 2, 'last': 26}]

        data = {'btc_usd': btc_usd, 'ltc_usd': ltc_usd}

        timestamps = merge_timestamps(data)

        self.assertEqual(timestamps, [1, 2, 3])

    def merge_timestamps_test_empty(self):
        result = merge_timestamps({})
        self.assertEqual(result, [])

    def merge_timestamps_test_2(self):
        btc_usd = [{'time': 1, 'last': 1}, {'time': 3, 'last': 2}]
        ltc_usd = []

        data = {'btc_usd': btc_usd, 'ltc_usd': ltc_usd}

        timestamps = merge_timestamps(data)

        self.assertEqual(timestamps, [1, 3])