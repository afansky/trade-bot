from matplotlib.backends.backend_svg import short_float_fmt
from pymongo import MongoClient
import pandas as pd
import datetime
import stockstats as ss
import numpy as np


def generate_signals(df, short_period, long_period):
    df_stock = ss.StockDataFrame.retype(df)
    df_stock['close_%s_ema_xu_close_%s_ema' % (short_period, long_period)]
    df_stock['close_%s_ema_xd_close_%s_ema' % (short_period, long_period)]




def load_data(ticker, short_period, long_period):
    print('Loading data...')

    client = MongoClient("mongodb://localhost:27017")
    db = client.bitcoinbot
    date_filter = {'$gte': datetime.datetime(2016, 6, 1), '$lt': datetime.datetime(2017, 3, 1)}
    ticks = db[ticker + '_1T'].find(
        {'time': date_filter}).sort([('time', 1)])
    df = pd.DataFrame.from_records(list(ticks), columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df = df.set_index(['time'])
    df = df.bfill()
    # df = df.dropna(0)

    print('Generating signals')
    generate_signals(df, short_period, long_period)

    print('Backtesting strategy on %s points' % len(df.index))

    usd = 10000
    bitcoin = 0
    trade_size = 500
    comission = 0.0026

    current_trades = []
    for i, c in enumerate(df.index):
        if i % 10000 == 0:
            print('Processing item #%s, usd: %s, bitcoin: %s' % (i, usd, bitcoin))

        row = df.iloc[i]
        close = row['close']

        if row['close_%s_ema_xd_close_%s_ema' % (short_period, long_period)]:
            if trade_size < usd:
                trade_volume = (trade_size / close) * (1 - comission)
                usd = usd - trade_size
                bitcoin = bitcoin + trade_volume
                current_trades.append(trade_volume)

                assert usd > 0
            else:
                print('Lost all shit.')
                break

        if row['close_%s_ema_xu_close_%s_ema' % (short_period, long_period)]:
            if current_trades:
                trade_volume = current_trades.pop()
                trade_price = trade_volume * close * (1 - comission)
                usd = usd + trade_price
                bitcoin = bitcoin - trade_volume

                assert bitcoin >= 0

    if bitcoin > 0:
        usd = usd + bitcoin * df.iloc[-1]['close'] * (1 - comission)
        bitcoin = 0

    print('Done backtesting. Current usd is %s and bitcoin is %s' % (usd, bitcoin))

    return usd


if __name__ == "__main__":
    results = []
    for short_period in range(3, 10):
        for long_period in range(15, 21):
            print('Backtesting strategy for short %s and long %s' % (short_period, long_period))

            usd = load_data('bitstampbtcusd', short_period, long_period)
            results.append([usd, short_period, long_period])

    print(results)