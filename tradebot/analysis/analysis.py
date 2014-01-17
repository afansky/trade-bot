import datetime
import indicator
import btcebot
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class Analyzer(object):
    indicators = [indicator.double_crossover]

    def __init__(self, database_path):
        self.database_path = database_path
        self.db = None

    def analyze(self, t, pair):
        ticks = self.get_db().retrieveTicks(pair, datetime.date.fromordinal(datetime.date.today().toordinal() - 1), t)
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
    index = []
    ticks_new = []
    for tick in ticks:
        index.append(tick[0])
        ticks_new.append(tick[4:])

    return pd.DataFrame(ticks_new, columns=['high_price', 'low_price', 'avg_price', 'last_price', 'buy_price', 'sell_price', 'volume', 'current_volume'], index=index)
