import threading
import btcebot


class DataSaver(object):
    def __init__(self, database_path):
        self.storage = threading.local()
        self.storage.db = None
        self.database_path = database_path

    def getDB(self):
        if getattr(self.storage, 'db', None) is None:
            self.storage.db = btcebot.MarketDatabase(self.database_path)

        return self.storage.db

    def closeDB(self):
        if self.storage.db is not None:
            self.storage.db.close()

    def saveDepth(self, t, pair, asks, bids):
        self.getDB().insertDepth(t, pair, asks, bids)

    def saveTradeHistory(self, new_trades):
        self.getDB().insertTradeHistory(new_trades)

    def saveTick(self, t, pair, tick):
        self.getDB().insertTick(t, pair, tick)