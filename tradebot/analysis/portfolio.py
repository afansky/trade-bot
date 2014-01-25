import datetime
import logging

logger = logging.getLogger(__name__)


class Portfolio(object):
    def __init__(self, portfolio=None):
        if portfolio:
            self.portfolio = portfolio
        else:
            self.portfolio = {}
        self.orders = {}

    def add_order(self, time, order):
        self.orders[time] = order

    def get_orders_to_process(self):
        time = datetime.datetime.now()
        orders = {k: v for k, v in self.orders.iteritems() if k < time}
        for k in orders.keys():
            if k in self.orders:
                del self.orders[k]
        return orders

    def amount_available(self, currency):
        if currency in self.portfolio:
            return self.portfolio[currency]
        return None

    def add_currency(self, currency, amount):
        if currency in self.portfolio:
            self.portfolio[currency] = self.portfolio[currency] + amount
        else:
            self.portfolio[currency] = amount


class Order(object):
    def __init__(self, currency, amount):
        self.currency = currency
        self.amount = amount


class SellOrder(Order):
    pass


class BuyOrder(Order):
    pass