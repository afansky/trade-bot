import datetime
import indicator
import btcebot
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def filter_repeating_ticks(ticks):
    result = []

    previous = None
    for t in ticks:
        if previous is None:
            previous = t
            continue

        if previous['last'] == t['last']:
            if previous['time'] != t['time']:
                previous['time'] = t['time']
            continue
        else:
            result.append(previous)

        previous = t

    if previous is not None:
        if len(result) == 0 or previous != result[-1]:
            result.append(previous)

    return result


class Analyzer(object):
    def __init__(self):
        self.db = None
        self.indicators = [indicator.double_crossover]
        self.last_price = None

    def analyze(self, t, pair):
        ticks = self.get_db().retrieve_ticks(pair, datetime.datetime.fromordinal(t.toordinal() - 1), t)

        ticks = filter_repeating_ticks(ticks)

        if len(ticks) < 35:
            logger.debug("no ticks found, aborting analysis")
            self.last_price = ticks[-1]['last']
            return

        last_price = ticks[-1]['last']
        if self.last_price == last_price:
            logger.debug("price has not changed since last iteration")
            return 

        df = create_data_frame(ticks)
        signals = []
        for func in self.indicators:
            signal = func(df)
            if signal is not None:
                signals.append(signal)

        for signal in signals:
            logger.info("Signal detected - %s @ %s - %s" % (signal.message, pair, signal.last_price))

        self.last_price = last_price

    def get_db(self):
        if self.db is None:
            self.db = btcebot.MarketDatabase()
        return self.db


def create_data_frame(ticks):
    return pd.DataFrame.from_records(ticks, index='time', columns=['time', 'high', 'low', 'avg', 'last', 'buy', 'sell',
                                                                   'vol', 'vol_cur'])