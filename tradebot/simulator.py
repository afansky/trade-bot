import logging
import database
import datetime
import analysis

logger = logging.getLogger(__name__)


class Simulator:
    def __init__(self, pairs):
        self.pairs = pairs
        self.db = database.MarketDatabase()
        self.analyzer = analysis.Analyzer()

    def simulate(self):
        for pair in self.pairs:
            logger.info("processing pair %s" % pair)
            ticks = self.db.retrieve_ticks(pair, datetime.datetime(2000, 1, 1), datetime.datetime.now())

            ticks = analysis.filter_repeating_ticks(ticks)

            total_len = len(ticks)
            logger.info("analyzing %s %s timestamps" % (pair, total_len))
            for i in range(1, total_len):
                if i % 1000 == 0:
                    logger.info('done %s percent for %s' % (i * 100 / total_len, pair))

                start = 1
                if i > 1000:
                    start = i - 1000
                data = ticks[start:i]
                self.analyzer.analyze(data, pair)





if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    pairs = ("btc_usd", "btc_rur", "btc_eur", "ltc_btc", "ltc_usd", "ltc_rur", "ltc_eur")
    simulator = Simulator(pairs)

    simulator.simulate()