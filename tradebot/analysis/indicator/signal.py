class BuySignal(object):
    def __init__(self, last_price=None):
        self.last_price = last_price
        self.message = "buy signal"


class SellSignal(object):
    def __init__(self, last_price=None):
        self.last_price = last_price
        self.message = "sell signal"