import unittest
import datetime
from analysis.portfolio import Portfolio, SellOrder, BuyOrder, NoFundsException


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
        self.assertEqual(orders[0], order)
        self.assertEqual(len(portfolio.orders), 0)

    def test_order_add_future(self):
        portfolio = Portfolio()
        order = SellOrder('usd', 50.0)

        portfolio.add_order(datetime.datetime.now() + datetime.timedelta(minutes=5), order)

        orders = portfolio.get_orders_to_process()
        self.assertEqual(len(orders), 0)
        self.assertEqual(len(portfolio.orders), 1)

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
        self.assertIn(order_1, orders)
        self.assertIn(order_2, orders)
        self.assertEqual(len(portfolio.orders), 0)

    def test_execute_order(self):
        portfolio = Portfolio({'usd': 100})

        portfolio.execute_order(BuyOrder('ltc_usd'), 25.0)

        self.assertEqual(portfolio.amount_available('ltc'), 4)

    def test_execute_order_no_money(self):
        portfolio = Portfolio()

        self.assertRaises(NoFundsException, lambda: portfolio.execute_order(BuyOrder('ltc_usd'), 25.0))

    def test_execute_order_no_money_2(self):
        portfolio = Portfolio({'usd': 100})

        self.assertRaises(NoFundsException, lambda: portfolio.execute_order(SellOrder('ltc_usd'), 25.0))

    def test_execute_order_multiple(self):
        portfolio = Portfolio({'usd': 100})

        portfolio.execute_order(BuyOrder('ltc_usd'), 25.0)
        self.assertEqual(portfolio.amount_available('ltc'), 4.0)
        self.assertEqual(portfolio.amount_available('usd'), 0.0)

        portfolio.execute_order(SellOrder('ltc_usd'), 30.0)
        self.assertEqual(portfolio.amount_available('usd'), 120.0)
        self.assertEqual(portfolio.amount_available('ltc'), 0.0)

