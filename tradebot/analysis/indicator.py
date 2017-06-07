import pandas as pd
import logging
import analysis
import numpy as np

logger = logging.getLogger(__name__)


def moving_average(df, period):
    return pd.rolling_mean(df, window=period, min_periods=0)


def exponential_moving_average(df, period):
    return pd.ewma(df, span=period)


def standard_deviation(df, period):
    return pd.rolling_std(df, window=period)


def double_crossover(df):
    df = df.resample("1Min", fill_method="ffill")

    ma5 = exponential_moving_average(df, 5)
    ma35 = exponential_moving_average(df, 35)

    previous_ma5_value = ma5.iloc[-2]['last']
    previous_ma35_value = ma35.iloc[-2]['last']

    current_ma5_value = ma5.iloc[-1]['last']
    current_ma35_value = ma35.iloc[-1]['last']

    last_price = df.iloc[-1]['last']

    if previous_ma5_value < previous_ma35_value and current_ma5_value >= current_ma35_value:
        return BuySignal(last_price, 'double_crossover')

    if previous_ma5_value > previous_ma35_value and current_ma5_value <= current_ma35_value:
        return SellSignal(last_price, 'double_crossover')


def normalize(df):
    if len(df.index) == 0:
        return df

    df_norm = df / df.iloc[0]['last']
    return df_norm

std_20 = None
ma_20 = None
bollinger = None


def bollinger_bands_value(df, time):
    global std_20
    if std_20 is None:
        std_20 = standard_deviation(df, 20)

    global ma_20
    if ma_20 is None:
        ma_20 = moving_average(df, 20)

    global bollinger
    if bollinger is None:
        bollinger = (df[20:] - ma_20[20:]) / std_20[20:]

    bollinger_value = bollinger.loc[time]['last']

    return bollinger_value


def bollinger_bands(df):
    global std_20
    if std_20 is None:
        std_20 = standard_deviation(df, 20)

    global ma_20
    if ma_20 is None:
        ma_20 = moving_average(df, 20)

    global bollinger
    if bollinger is None:
        bollinger = (df[20:] - ma_20[20:]) / std_20[20:]

    return bollinger


def macd(df):
    ema_26 = df.ewm(span=26, min_periods=26).mean().bfill()
    ema_12 = df.ewm(span=12, min_periods=12).mean().bfill()
    # ema_9 = df.ewm(span=9, min_periods=9).mean().bfill()
    macd_value = ema_26 - ema_12
    # result = pd.concat([macd_value, ema_9], axis=1)
    macd_value.columns = ['macd']
    return macd_value

def rsi(df):
    rsi_delta = df.diff().bfill()
    delta_up = rsi_delta.copy()
    delta_down = rsi_delta.copy()
    delta_up[delta_up < 0] = 0
    delta_down[delta_down > 0] = 0
    average_gain = pd.rolling_mean(delta_up, 14).bfill()
    average_loss = pd.rolling_mean(delta_down, 14).bfill().abs()

    rs_value = average_gain / average_loss
    rs_value = rs_value.fillna(method='bfill')
    rsi_result = 100.0 - 100.0 / (1.0 + rs_value)
    rsi_result = rsi_result.bfill()
    return rsi_result


def rsi_value(df, time):
    global rsi_result
    if rsi_result is None:
        rsi_delta = df.diff()
        delta_up = rsi_delta.copy()
        delta_down = rsi_delta.copy()
        delta_up[delta_up < 0] = 0
        delta_down[delta_down > 0] = 0
        average_gain = pd.rolling_mean(delta_up, 14)
        average_loss = pd.rolling_mean(delta_down, 14).abs()

        rs_value = average_gain / average_loss
        rsi_result = 100.0 - 100.0 / (1.0 + rs_value)
    return rsi_result.loc[time]['last']


def stoch_rsi(df, period):
    rsi_value = rsi(df, period=period)

    stoch_rsi_value = (rsi_value - pd.rolling_min(rsi_value, period)) / \
                      (pd.rolling_max(rsi_value, period) - pd.rolling_min(rsi_value, period))

    return stoch_rsi_value


class BuySignal(object):
    def __init__(self, last_price=None, source=None):
        self.last_price = last_price
        self.message = "buy signal"
        self.source = source


class SellSignal(object):
    def __init__(self, last_price=None, source=None):
        self.last_price = last_price
        self.message = "sell signal"
        self.source = source