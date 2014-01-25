class BuySignal(object):
    def __init__(self, last_price=None, source=None):
        self.last_price = last_price
        self.message = "buy signal"
        self.source = source


class SellSignal(object):
    def __init__(self, last_price=None, source=None):
        self.last_price = last_price
        self.message = "sell signal"
        self.source = source