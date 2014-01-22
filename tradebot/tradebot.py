import time
import btcebot
import saver
import analysis
import logging

logger = logging.getLogger(__name__)


class TradeBot(btcebot.TraderBase):
    def __init__(self, pairs, database_path, disable_saver):
        btcebot.TraderBase.__init__(self, pairs)
        self.trade_history_seen = {}
        self.saver = saver.DataSaver(database_path, disable_saver)
        self.analyzer = analysis.Analyzer(database_path)

    def onExit(self):
        self.saver.close_db()

    def onNewDepth(self, t, pair, asks, bids):
        self.saver.save_depth(t, pair, asks, bids)

        self.analyzer.analyze(t, pair)

    def onNewTradeHistory(self, t, pair, trades):
        history = self.trade_history_seen.setdefault(pair, set())

        new_trades = filter(lambda trade: trade.tid not in history, trades)
        if new_trades:
            self.saver.save_trade_history(new_trades)
            history.update(t.tid for t in new_trades)

    def onNewTicker(self, t, pair, ticker):
        self.saver.save_tick(t, pair, ticker)


def on_bot_error(msg, tracebackText):
    tstr = time.strftime("%Y/%m/%d %H:%M:%S")
    # logger.error("%s - %s\n%s\n%s\n" % (tstr, msg, tracebackText, "-"*80))
    open("logger-bot-error.log", "a").write(
        "%s - %s\n%s\n%s\n" % (tstr, msg, tracebackText, "-"*80))

def run(database_path, disable_saver):
    pairs = ("btc_usd", "btc_rur", "btc_eur", "ltc_btc", "ltc_usd",
             "ltc_rur", "ltc_eur")
    botlogger = TradeBot(pairs, database_path, disable_saver)
    #logger= MarketDataLogger(("btc_usd", "ltc_usd"), database_path)

    # Create a bot and add the logger to it.
    bot = btcebot.Bot()
    bot.addTrader(botlogger)

    # Add an error handler so we can print info about any failures
    bot.addErrorHandler(on_bot_error)

    # The bot will provide the logger with updated information every
    # 60 seconds.
    bot.setCollectionInterval(60)
    bot.tickerInterval = 60
    bot.start()
    logger.info("Running; press Ctrl-C to stop")

    try:
        while 1:
            # you can do anything else you prefer in this loop while
            # the bot is running in the background
            time.sleep(3600)

    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        bot.stop()


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    import argparse
    parser = argparse.ArgumentParser(description='Simple range trader example.')
    parser.add_argument('--db-path', default='btce.sqlite',
                        help='Path to the logger database.')
    parser.add_argument('--disable-saver', default=True)

    args = parser.parse_args()
    run(args.db_path, args.disable_saver)
