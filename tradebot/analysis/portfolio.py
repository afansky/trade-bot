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

    def execute_order(self, order, current_price):
        base, counter = order.pair.split('_')
        if isinstance(order, BuyOrder):
            available = self.amount_available(counter)
            if available > 0:
                buy_amount = available / current_price
                self.add_currency(base, buy_amount)
                self.add_currency(counter, -available)
            else:
                raise NoFundsException
        else:
            available = self.amount_available(base)
            if available > 0:
                sell_amount = available * current_price
                self.add_currency(counter, sell_amount)
                self.add_currency(base, -available)
            else:
                raise NoFundsException


class Order(object):
    def __init__(self, pair, amount=None):
        self.pair = pair
        self.amount = amount

    def get_type(self):
        pass


class SellOrder(Order):
    def get_type(self):
        return "sell"


class BuyOrder(Order):
    def get_type(self):
        return "buy"


class NoFundsException(Exception):
    pass