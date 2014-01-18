import datetime
import indicator
import btcebot
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class Analyzer(object):
    def __init__(self, database_path):
        self.database_path = database_path
        self.db = None
        self.indicators = [indicator.double_crossover]

    def analyze(self, t, pair):
        ticks = self.get_db().retrieveTicks(pair, datetime.datetime.fromordinal(datetime.datetime.today().toordinal() - 1), t)

        if len(ticks) == 0:
            logger.debug("no ticks found, aborting analysis")
            return

        df = create_data_frame(ticks)
        signals = []
        for func in self.indicators:
            signal = func(df)
            if signal is not None:
                signals.append(signal)

        for signal in signals:
            logger.info("Signal detected - %s @ %s" % (signal.message, pair))

    def get_db(self):
        if self.db is None:
            self.db = btcebot.MarketDatabase(self.database_path)
        return self.db


def create_data_frame(ticks):
    return pd.DataFrame.from_records(ticks, index='time', columns=['time', 'high', 'low', 'avg', 'last', 'buy', 'sell',
                                                                   'vol', 'vol_cur'])