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
            timestamps = self.db.retrieve_ticks_timestamps(pair)

            total_len = len(timestamps)
            logger.info("analyzing %s %s timestamps" % (pair, total_len))
            count = 0
            for timestamp in timestamps:
                count += 1
                if count % 1000 == 0:
                    logger.info('done %s percent for %s' % (count * 100 / total_len, pair))

                self.analyzer.analyze(timestamp['time'], pair)





if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    pairs = ("btc_usd", "btc_rur", "btc_eur", "ltc_btc", "ltc_usd", "ltc_rur", "ltc_eur")
    simulator = Simulator(pairs)

    simulator.simulate()