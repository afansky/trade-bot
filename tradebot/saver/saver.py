import btcebot


class DataSaver(object):
    def __init__(self, database_path):
        self.db = None
        self.database_path = database_path

    def getDB(self):
        if self.db is None:
            self.db = btcebot.MarketDatabase(self.database_path)

        return self.db

    def closeDB(self):
        if self.db is not None:
            self.db.close()

    def saveDepth(self, t, pair, asks, bids):
        self.getDB().insertDepth(t, pair, asks, bids)

    def saveTradeHistory(self, new_trades):
        self.getDB().insertTradeHistory(new_trades)

    def saveTick(self, t, pair, tick):
        self.getDB().insertTick(t, pair, tick)
        pass