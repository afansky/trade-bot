import unittest
from analysis import create_data_frame
from indicator import normalize
from pandas.util.testing import assert_frame_equal


class TestIndicatorFunctions(unittest.TestCase):

    def normalize_test(self):
        data = [{'time': 1, 'last': 10}, {'time': 2, 'last': 20}]
        df = create_data_frame(data)

        df = normalize(df)

        assert_frame_equal(df, create_data_frame([{'time': 1, 'last': 1}, {'time': 2, 'last': 2}]))

    def normalize_test_empty(self):
        data = []
        df = create_data_frame(data)

        df = normalize(df)

        assert_frame_equal(df, create_data_frame([]))

    def normalize_test_2(self):
        data = [{'time': 1, 'last': 10.0}, {'time': 2, 'last': 15.0}, {'time': 3, 'last': 7.0}, {'time': 4, 'last': 9.0},
                {'time': 5, 'last': 11.0}]
        df = create_data_frame(data)

        df = normalize(df)

        assert_frame_equal(df, create_data_frame([{'time': 1, 'last': 1.0}, {'time': 2, 'last': 1.5}, {'time': 3, 'last': 0.7},
                                                  {'time': 4, 'last': 0.9}, {'time': 5, 'last': 1.1}]))

