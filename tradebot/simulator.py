import logging
import database
import datetime
import analysis
from analysis.indicator.signal import BuySignal
from analysis.portfolio import Portfolio, BuyOrder, SellOrder

logger = logging.getLogger(__name__)


def merge_timestamps(data):
    result = []
    for val in data.values():
        timestamps = [item['time'] for item in val]
        result = result + list(set(timestamps) - set(result))
    return result


class Simulator:
    def __init__(self, pairs, initial_portfolio=None):
        self.pairs = pairs
        self.db = database.MarketDatabase()
        self.analyzer = analysis.Analyzer()
        self.portfolio = Portfolio(initial_portfolio)

    def simulate(self):
        data = {}
        for pair in self.pairs:
            logger.info("loading data for %s" % pair)
            ticks = self.db.retrieve_ticks(pair, datetime.datetime(2000, 1, 1), datetime.datetime.now())
            ticks = analysis.filter_repeating_ticks(ticks)
            data[pair] = ticks

        logger.info("all data loaded")

        timestamps = merge_timestamps(data)



            # total_len = len(ticks)
            # logger.info("analyzing %s %s timestamps" % (pair, total_len))
            # first_cur, second_cur = pair.split('_')
            # for i in range(1, total_len):
            #     if i % 1000 == 0:
            #         logger.info('done %s percent for %s' % (i * 100 / total_len, pair))
            #
            #     start = 1
            #     if i > 1000:
            #         start = i - 1000
            #     data = ticks[start:i]
            #
            #     signals = self.analyzer.analyze(data, pair)
            #     for signal in signals:
            #         if isinstance(signal, BuySignal):
            #             if self.portfolio.amount_available(second_cur):
            #                 self.portfolio.add_order(datetime.datetime.now(), BuyOrder('btc', 5))
            #                 self.portfolio.add_order(datetime.datetime.now() + datetime.timedelta(minutes=5), SellOrder('btc', 5))






if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    pairs = ("btc_usd", "btc_rur", "btc_eur", "ltc_btc", "ltc_usd", "ltc_rur", "ltc_eur")
    simulator = Simulator(pairs)

    simulator.simulate()