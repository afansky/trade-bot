import pandas as pd
import logging
import signal

logger = logging.getLogger(__name__)


def moving_average(df, period):
    return pd.rolling_mean(df.resample("1Min", fill_method="ffill"), window=period, min_periods=0)


def exponential_moving_average(df, period):
    return pd.ewma(df.resample("1Min", fill_method="ffill"), span=period)


def double_crossover(df):
    data_size = len(df.index)
    if data_size < 35:
        logger.info("skipping iteration, not enough data - %s rows in the data frame" % data_size)
        return

    ma5 = exponential_moving_average(df, 5)
    ma35 = exponential_moving_average(df, 35)

    previous_ma5_value = ma5.iloc[-2]['last']
    previous_ma35_value = ma35.iloc[-2]['last']

    current_ma5_value = ma5.iloc[-1]['last']
    current_ma35_value = ma35.iloc[-1]['last']

    last_price = df.iloc[-1]['last']

    if previous_ma5_value < previous_ma35_value and current_ma5_value >= current_ma35_value:
        return signal.BuySignal(last_price)

    if previous_ma5_value > previous_ma35_value and current_ma5_value <= current_ma35_value:
        return signal.SellSignal(last_price)
