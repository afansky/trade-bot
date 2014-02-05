import pandas as pd
import logging
import signal

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
        return signal.BuySignal(last_price, 'double_crossover')

    if previous_ma5_value > previous_ma35_value and current_ma5_value <= current_ma35_value:
        return signal.SellSignal(last_price, 'double_crossover')


def normalize(df):
    if len(df.index) == 0:
        return df

    df_norm = df / df.iloc[0]['last']
    return df_norm


def bollinger_bands(df):
    std_20 = standard_deviation(df, 20)
    ma_20 = moving_average(df, 20)

    bollinger = (df - ma_20) / std_20

    bollinger_value_previous = bollinger.iloc[-2]['last']
    bollinger_value = bollinger.iloc[-1]['last']

    last_price = df.iloc[-1]['last']

    if bollinger_value <= -2.0 <= bollinger_value_previous:
        return signal.BuySignal(last_price, 'bollinger_bands')


def rsi(df, period):
    delta = df.diff()

    delta_up = delta.copy()
    delta_down = delta.copy()
    delta_up[delta_up < 0] = 0
    delta_down[delta_down > 0] = 0

    average_gain = pd.rolling_mean(delta_up, period)
    average_loss = pd.rolling_mean(delta_down, period).abs()

    rs_value = average_gain / average_loss
    return 100.0 - 100.0 / (1.0 + rs_value)


def stoch_rsi(df, period):
    rsi_value = rsi(df, period=period)

    stoch_rsi_value = (rsi_value - pd.rolling_min(rsi_value, period)) / \
                      (pd.rolling_max(rsi_value, period) - pd.rolling_min(rsi_value, period))

    return stoch_rsi_value