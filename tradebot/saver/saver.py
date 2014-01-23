import threading
import btcebot
import logging

logger = logging.getLogger(__name__)


class DataSaver(object):
    def __init__(self, database_path, disable_saver):
        self.enabled = disable_saver is not True
        if self.enabled:
            logger.info("initializing saver")
            self.storage = threading.local()
            self.storage.db = None
            self.database_path = database_path
        else:
            logger.info("saver is disabled")

    def get_db(self):
        if not self.enabled:
            return

        if getattr(self.storage, 'db', None) is None:
            self.storage.db = btcebot.MarketDatabase()

        return self.storage.db

    def close_db(self):
        if not self.enabled:
            return

        if self.storage.db is not None:
            self.storage.db.close()

    def save_depth(self, t, pair, asks, bids):
        if not self.enabled:
            return

        self.get_db().insert_depth(t, pair, asks, bids)

    def save_trade_history(self, new_trades):
        if not self.enabled:
            return

        self.get_db().insert_trade_history(new_trades)

    def save_tick(self, t, pair, tick):
        if not self.enabled:
            return

        self.get_db().insert_tick(t, pair, tick)