import pandas as pd
import logging
import signal

logger = logging.getLogger('indicator')


def moving_average(df, period):
    return pd.rolling_mean(df.resample("1Min", fill_method="ffill"), window=period, min_periods=0)


def exponential_moving_average(df, period):
    return pd.ewma(df, span=period)


def double_crossover(df):
    ma5 = exponential_moving_average(df, 5)
    ma35 = exponential_moving_average(df, 35)

    previous_ma5_value = ma5.iat[-2, 0]
    previous_ma35_value = ma35.iat[-2, 0]

    current_ma5_value = ma5.iat[-1, 0]
    current_ma35_value = ma35.iat[-1, 0]

    if previous_ma5_value < previous_ma35_value and current_ma5_value >= current_ma35_value:
        return signal.BuySignal()

    if previous_ma5_value > previous_ma35_value and current_ma5_value <= current_ma35_value:
        return signal.SellSignal()
