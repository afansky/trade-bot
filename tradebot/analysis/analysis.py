import indicator

class Analyzer(object):

    def __init__(self, database_path):
        self.database_path = database_path

    def analyze(self, pair, asks, bids):
        ma = indicator.moving_average(pair, asks, bids)
