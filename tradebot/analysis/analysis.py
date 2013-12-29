import datetime
import indicator
import btcebot
import logging
import pandas as pd

logging.basicConfig()
logger = logging.getLogger('analysis')
logger.setLevel(logging.INFO)


class Analyzer(object):

    indicators = [indicator.double_crossover]

    def __init__(self, database_path):
        self.database_path = database_path
        self.db = None

    def analyze(self, t, pair, asks, bids):
        ticks = self.get_db().retrieveTicks(pair, datetime.date(2013, 12, 23), t)

        df = self.create_data_frame(ticks)
        signals = []
        for func in self.indicators:
            signal = func(df)
            if signal is not None:
                signals.append(signal)

        for signal in signals:
            logger.info("signal detected")

    def get_db(self):
        if self.db is None:
            self.db = btcebot.MarketDatabase(self.database_path)
        return self.db

    def create_data_frame(self, ticks):
        index = []
        ticks_new = []
        for tick in ticks:
            index.append(tick[0])
            ticks_new.append(tick[2:])

        return pd.DataFrame(ticks_new, columns=['ask_price', 'ask_volume', 'bid_price', 'bid_volume'], index=index)
