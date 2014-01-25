import unittest
import datetime
from portfolio import Portfolio, SellOrder, BuyOrder


class TestPortfolioFunctions(unittest.TestCase):

    def test_portfolio_create(self):
        portfolio = Portfolio({'usd': 100.0})
        self.assertEqual(portfolio.amount_available('usd'), 100.0)

    def test_portfolio_add(self):
        portfolio = Portfolio({'usd': 100.0})
        portfolio.add_currency('usd', 50.0)
        self.assertEqual(portfolio.amount_available('usd'), 150.0)

    def test_portfolio_add_new(self):
        portfolio = Portfolio({'usd': 100.0})
        portfolio.add_currency('eur', 50.0)
        self.assertEqual(portfolio.amount_available('eur'), 50.0)

    def test_portfolio_subtract(self):
        portfolio = Portfolio({'usd': 100.0})
        portfolio.add_currency('usd', -50.0)
        self.assertEqual(portfolio.amount_available('usd'), 50.0)

    def test_order_add(self):
        portfolio = Portfolio()
        order = SellOrder('usd', 50.0)

        portfolio.add_order(datetime.datetime.now(), order)

        orders = portfolio.get_orders_to_process()
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders.itervalues().next(), order)
        self.assertEqual(len(portfolio.orders), 0)

    def test_order_add_multiple(self):
        portfolio = Portfolio()
        order_1 = SellOrder('usd', 50.0)
        order_2 = BuyOrder('usd', 100.0)

        time_1 = datetime.datetime.now()
        portfolio.add_order(time_1, order_1)

        time_2 = datetime.datetime.now()
        portfolio.add_order(time_2, order_2)

        orders = portfolio.get_orders_to_process()
        self.assertEqual(len(orders), 2)
        self.assertEqual(orders[time_1], order_1)
        self.assertEqual(orders[time_2], order_2)
        self.assertEqual(len(portfolio.orders), 0)
