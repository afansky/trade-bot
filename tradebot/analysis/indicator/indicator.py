import pandas as pd
import logging

logger = logging.getLogger('indicator')


def moving_average(ticks):
    tick_list = []
    index = []
    for tick in ticks:
        tick_list.append(tick[1:])
        index.append(tick[0])

    df = pd.DataFrame(tick_list, columns=['pair', 'ask_price', 'ask_volume', 'bid_price', 'bid_volume'], index=index)

    ma = pd.rolling_mean(df.resample("1H", fill_method="ffill"), window=3, min_periods=1)

    return ma.iat[-1,0]