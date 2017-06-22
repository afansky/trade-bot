# Copyright (c) 2013 Alan McIntyre

import decimal
from pymongo import MongoClient
import csv
import pandas as pd
from datetime import datetime



# Add support for conversion to/from decimal
def adapt_decimal(d):
    return int(d * decimal.Decimal("1e8"))


def convert_decimal(s):
    return decimal.Decimal(s) * decimal.Decimal("1e-8")


class MarketDatabase(object):
    def __init__(self):
        self.client = MongoClient('mongodb://192.168.1.102:27017/')
        self.db = self.client.tradebot

    def close(self):
        self.client.close()

    def insert_trade_history(self, trade_data):
        pass

    def retrieve_trade_history(self, start_date, end_date, pair):
        pass

    def insert_tick(self, time, pair, tick):
        tick_hash = {k: lambda v: v if not isinstance(v, decimal.Decimal) else str(v) for k, v in tick.items()}
        tick_hash["time"] = time

        tick_hash = {"time": time, "pair": pair, "updated": tick.updated, "server_time": tick.server_time,
                     "high": float(tick.high), "low": float(tick.low), "avg": float(tick.avg), "last": float(tick.last),
                     "buy": float(tick.buy), "sell": float(tick.sell), "vol": float(tick.vol),
                     "vol_cur": float(tick.vol_cur)}
        # self.db.ticks.insert(tick_hash)

    def retrieve_ticks(self, pair, start_time, end_time):
        ticks = self.db.ticks.find({"pair": pair, "time": {"$gte": start_time, "$lte": end_time}},
                                   {'time': 1, 'last': 1}).sort('time', 1)
        return list(ticks)

    def retrieve_ticks_timestamps(self, pair):
        timestamps = self.db.ticks.find({'pair': pair}, {'time': 1}).sort('time', 1)
        return list(timestamps)

    def insert_depth(self, dt, pair, asks, bids):
        pass

    def retrieve_depth(self, start_date, end_date, pair):
        pass


class CsvDatabase(object):
    def __init__(self):
        pass

    def retrieve_ticks(self, pair, start_time, end_time):
        df = pd.read_csv('../../data/large.csv', header=None, sep=",", names=['time', 'last', 'volume'],
                         parse_dates=True, date_parser=dateparse, index_col=0)
        df['last'] = pd.to_numeric(df['last'])
        # df = df.groupby('time').last()
        df = df.resample('30T').last().ffill()
        size = len(df.index)
        train = df.head(int(size * 0.6))

        test = df.tail(int(size * 0.4))
        cv = test.tail(int(size * 0.2))
        test = test.head(int(size * 0.2))
        learn = test.tail(int(len(test.index) * 0.57))

        return train, cv, test, learn

    def load_all_data(self):
        df = pd.read_csv('../../data/large.csv', header=None, sep=",", names=['time', 'last', 'volume'],
                         parse_dates=True, date_parser=dateparse, index_col=0)
        df['last'] = pd.to_numeric(df['last'])
        df = df.resample('8H').last().ffill()
        return df

    def import_data_to_db(self):
        df = pd.read_csv('../../data/btceusd_last.csv', header=None, sep=",", names=['time', 'last', 'volume'],
                         parse_dates=True, date_parser=dateparse, index_col=0)
        df['last'] = pd.to_numeric(df['last']).ffill()
        client = MongoClient("mongodb://localhost:27017")
        db = client.bitcoinbot
        counter = 0

        for i, row in df.iterrows():
            counter = counter + 1
            if counter % 20000 == 0:
                print('Processing tick number %s', counter)
            if counter < 17329713:
                continue
            db.btcebtcusd.insert_one({'time': i, 'last': row['last'], 'volume': row['volume']})

    def import_resampled_data(self, period):
        print('Loading CSV file...')
        df = pd.read_csv('../../data/btceusd_last.csv', header=None, sep=",", names=['time', 'last', 'volume'],
                         parse_dates=True, date_parser=dateparse, index_col=0)
        print('Resampling data...')
        df['last'] = pd.to_numeric(df['last']).bfill()
        df['volume'] = pd.to_numeric(df['volume']).bfill()
        price = df.resample(period, how={'last': 'ohlc'})
        volume = df.resample(period, how={'volume': 'sum'})
        volume.columns = pd.MultiIndex.from_tuples([('volume', 'sum')])
        df = pd.concat([price, volume], axis=1)

        print('Uploading to the database...')
        client = MongoClient("mongodb://localhost:27017")
        db = client.bitcoinbot
        counter = 0

        for i, row in df.iterrows():
            counter = counter + 1
            if counter % 20000 == 0:
                print('Processing tick number %s', counter)
            db.btcebtcusd_3d.insert_one(
                {'time': i, 'open': row['last']['open'], 'high': row['last']['high'], 'low': row['last']['low'],
                 'close': row['last']['close'], 'volume': row['volume']['sum']})

        print('Done.')


    def load_samples(self):
        df = pd.read_csv('../../data/last_price_2.csv', header=None, sep=",", names=['time', 'last', 'volume'],
                         parse_dates=True, date_parser=dateparse, index_col=0)
        df['last'] = pd.to_numeric(df['last'])
        df = df.resample('1H').last().ffill()
        return df

def dateparse(time_in_secs):
    return datetime.fromtimestamp(float(time_in_secs))