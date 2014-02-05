import logging
import database
import datetime
import analysis.analysis
from analysis.indicator import BuySignal
from analysis.portfolio import Portfolio, BuyOrder, SellOrder, NoFundsException

logger = logging.getLogger(__name__)


def merge_timestamps(data):
    result = []
    for val in data.values():
        timestamps = [item['time'] for item in val]
        result = result + list(set(timestamps) - set(result))
    result.sort()
    return result


class Simulator:
    def __init__(self, pairs, initial_portfolio=None):
        self.pairs = pairs
        self.db = database.MarketDatabase()
        self.analyzer = analysis.analysis.Analyzer()
        self.portfolio = Portfolio(initial_portfolio)
        self.last_prices = {}

    def simulate(self):
        data = {}
        for pair in self.pairs:
            logger.info("loading data for %s" % pair)
            ticks = self.db.retrieve_ticks(pair, datetime.datetime(2000, 1, 1), datetime.datetime.now())
            ticks = analysis.analysis.filter_repeating_ticks(ticks)
            data[pair] = ticks

        logger.info("all data loaded")

        timestamps = merge_timestamps(data)
        logger.info("timestamps merged")

        for (i, t) in enumerate(timestamps):
            for pair in self.pairs:
                if not t in [tick['time'] for tick in data[pair]]:
                    continue

                start = 0
                if i > 1000:
                    start = i - 1000
                ticks = data[pair][start:i]

                signals = self.analyzer.analyze(ticks, pair)

                base, counter = pair.split('_')

                self.last_prices[pair] = data[pair][i]['last']

                if signals:
                    for signal in signals:
                        if isinstance(signal, BuySignal):
                            if self.portfolio.amount_available(counter):
                                self.portfolio.add_order(t, BuyOrder(pair))
                                self.portfolio.add_order(t + datetime.timedelta(minutes=5), SellOrder(pair))

                for order in self.portfolio.get_orders_to_process(t):
                    try:
                        order_base, order_counter = order.pair.split('_')
                        self.portfolio.execute_order(order, self.last_prices[order.pair])
                        logger.info("processed %s order %s @ %s - %s and %s now at portfolio for %s" %
                                    (order.get_type(), order.pair, self.last_prices[order.pair], self.portfolio.amount_available(order_base),
                                     self.portfolio.amount_available(order_counter), order.pair))
                    except NoFundsException:
                        logger.debug("can't process %s order for %s" % (order.get_type(), pair))


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    # pairs = ("btc_usd", "btc_rur", "btc_eur", "ltc_btc", "ltc_usd", "ltc_rur", "ltc_eur")
    pairs = ('btc_usd',)
    simulator = Simulator(pairs, {'usd': 1000, 'btc': 0, 'ltc': 0})

    simulator.simulate()