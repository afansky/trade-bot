import unittest
import analysis.indicator as i
from analysis.analysis import create_data_frame
import mock
from pandas.util.testing import assert_frame_equal


class TestIndicatorFunctions(unittest.TestCase):

    def normalize_test(self):
        data = [{'time': 1, 'last': 10}, {'time': 2, 'last': 20}]
        df = create_data_frame(data)

        df = i.normalize(df)

        assert_frame_equal(df, create_data_frame([{'time': 1, 'last': 1.0}, {'time': 2, 'last': 2.0}]))

    def normalize_test_empty(self):
        data = []
        df = create_data_frame(data)

        df = i.normalize(df)

        assert_frame_equal(df, create_data_frame([]))

    def normalize_test_2(self):
        data = [{'time': 1, 'last': 10.0}, {'time': 2, 'last': 15.0}, {'time': 3, 'last': 7.0}, {'time': 4, 'last': 9.0},
                {'time': 5, 'last': 11.0}]
        df = create_data_frame(data)

        df = i.normalize(df)

        assert_frame_equal(df, create_data_frame([{'time': 1, 'last': 1.0}, {'time': 2, 'last': 1.5}, {'time': 3, 'last': 0.7},
                                                  {'time': 4, 'last': 0.9}, {'time': 5, 'last': 1.1}]))

    def rsi_test(self):
        data = [{'time': 1, 'last': 44.34}, {'time': 2, 'last': 44.09}, {'time': 3, 'last': 44.15},
                {'time': 4, 'last': 43.61}, {'time': 5, 'last': 44.33}, {'time': 6, 'last': 44.83},
                {'time': 7, 'last': 45.10}, {'time': 8, 'last': 45.42}, {'time': 9, 'last': 45.84},
                {'time': 10, 'last': 46.08}, {'time': 11, 'last': 45.89}, {'time': 12, 'last': 46.03},
                {'time': 13, 'last': 45.61}, {'time': 14, 'last': 46.28}, {'time': 15, 'last': 46.28},
                {'time': 16, 'last': 46.00}, {'time': 17, 'last': 46.03}, {'time': 18, 'last': 46.41},
                {'time': 19, 'last': 46.22}, {'time': 20, 'last': 45.64}]
        df = create_data_frame(data)

        rsi_value = i.rsi(df, period=14)

        result = [{'time': 1, 'last': None}, {'time': 2, 'last': None}, {'time': 3, 'last': None},
                  {'time': 4, 'last': None}, {'time': 5, 'last': None}, {'time': 6, 'last': None},
                  {'time': 7, 'last': None}, {'time': 8, 'last': None}, {'time': 9, 'last': None},
                  {'time': 10, 'last': None}, {'time': 11, 'last': None}, {'time': 12, 'last': None},
                  {'time': 13, 'last': None}, {'time': 14, 'last': None}, {'time': 15, 'last': 70.46416},
                  {'time': 16, 'last': 70.02096}, {'time': 17, 'last': 69.83122}, {'time': 18, 'last': 80.56769},
                  {'time': 19, 'last': 73.33333}, {'time': 20, 'last': 59.80630}]
        result_df = create_data_frame(result)
        assert_frame_equal(rsi_value, result_df)

    def stoch_rsi_test(self):
        data = [{'time': 1, 'last': 54.09}, {'time': 2, 'last': 59.90}, {'time': 3, 'last': 58.20},
                {'time': 4, 'last': 59.76}, {'time': 5, 'last': 52.35}, {'time': 6, 'last': 52.82},
                {'time': 7, 'last': 56.94}, {'time': 8, 'last': 57.47}, {'time': 9, 'last': 55.26},
                {'time': 10, 'last': 57.51}, {'time': 11, 'last': 54.80}, {'time': 12, 'last': 51.47},
                {'time': 13, 'last': 56.16}, {'time': 14, 'last': 58.34}, {'time': 15, 'last': 56.02},
                {'time': 16, 'last': 60.22}, {'time': 17, 'last': 56.75}, {'time': 18, 'last': 57.38},
                {'time': 19, 'last': 50.23}, {'time': 20, 'last': 57.06}]
        df = create_data_frame(data)

        i.rsi = mock.Mock(return_value=df)
        stoch_rsi_value = i.stoch_rsi(df, period=14)

        result = [{'time': 1, 'last': None}, {'time': 2, 'last': None}, {'time': 3, 'last': None},
                  {'time': 4, 'last': None}, {'time': 5, 'last': None}, {'time': 6, 'last': None},
                  {'time': 7, 'last': None}, {'time': 8, 'last': None}, {'time': 9, 'last': None},
                  {'time': 10, 'last': None}, {'time': 11, 'last': None}, {'time': 12, 'last': None},
                  {'time': 13, 'last': None}, {'time': 14, 'last': 0.81495}, {'time': 15, 'last': 0.53974},
                  {'time': 16, 'last': 1.0}, {'time': 17, 'last': 0.60343}, {'time': 18, 'last': 0.67543},
                  {'time': 19, 'last': 0.0}, {'time': 20, 'last': 0.683681}]
        result_df = create_data_frame(result)
        assert_frame_equal(stoch_rsi_value, result_df)
