import datetime
import indicator
import btcebot
import logging
logging.basicConfig()
logger = logging.getLogger('analysis')
logger.setLevel(logging.INFO)

class Analyzer(object):

    def __init__(self, database_path):
        self.database_path = database_path
        self.db = None

    def analyze(self, t, pair, asks, bids):
        if pair == "ltc_usd":
            ticks = self.get_db().retrieveTicks(pair, datetime.date(2013, 12, 23), t)
            ma = indicator.moving_average(ticks)
            logger.info("Moving average for LTC/USD: %s" % ma)

    def get_db(self):
        if self.db is None:
            self.db = btcebot.MarketDatabase(self.database_path)
        return self.db