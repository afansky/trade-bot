import logging
from analysis.portfolio import Portfolio
import database
from analysis.analysis import filter_repeating_ticks, Analyzer
import datetime
from simulator import merge_timestamps

logger = logging.getLogger(__name__)


class EventProfiler:
    def __init__(self, trade_pair, initial_portfolio=None):
        self.pair = trade_pair
        self.db = database.CsvDatabase()
        self.analyzer = Analyzer()
        self.portfolio = Portfolio(initial_portfolio)
        self.signal_count = 0

    def profile(self):
        logger.info("loading data for %s" % self.pair)
        data = self.db.retrieve_ticks(self.pair, datetime.datetime(2000, 1, 1), datetime.datetime.now())

        # print(data)

        # data = filter_repeating_ticks(data)

        logger.info("all data loaded")

        logger.info("timestamps merged")

        for (i, t) in enumerate(data.index.values):
            if i > 40:
                start = i - 40
                ticks = data[start:i]

                signals = self.analyzer.analyze(ticks, pair)

                if signals:
                    self.signal_count += len(signals)

        logger.info("found %s signals" % self.signal_count)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    logger.info("starting event profiler")
    start_date = datetime.datetime.now()

    # pairs = ("btc_usd", "btc_rur", "btc_eur", "ltc_btc", "ltc_usd", "ltc_rur", "ltc_eur")
    pair = 'btc_usd'
    profiler = EventProfiler(pair)

    profiler.profile()

    end_date = datetime.datetime.now()
    logger.info("finished event profiler process, it took %s seconds" % (end_date - start_date).seconds)