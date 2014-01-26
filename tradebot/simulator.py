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
    result.sort()
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
                # for signal in signals:
                #     if isinstance(signal, BuySignal):
                #         if self.portfolio.amount_available(second_cur):
                #             self.portfolio.add_order(datetime.datetime.now(), BuyOrder('btc', 5))
                #             self.portfolio.add_order(datetime.datetime.now() + datetime.timedelta(minutes=5), SellOrder('btc', 5))






if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    # pairs = ("btc_usd", "btc_rur", "btc_eur", "ltc_btc", "ltc_usd", "ltc_rur", "ltc_eur")
    pairs = ('btc_usd',)
    simulator = Simulator(pairs)

    simulator.simulate()