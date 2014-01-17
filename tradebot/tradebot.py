import time
import btceapi
import btcebot
import saver
import analysis
import logging

logger = logging.getLogger(__name__)

class TradeBot(btcebot.TraderBase):
    '''
    This "trader" simply logs all of the updates it receives from the bot.
    '''

    def __init__(self, pairs, database_path):
        btcebot.TraderBase.__init__(self, pairs)
        self.trade_history_seen = {}
        self.saver = saver.DataSaver(database_path)
        self.analyzer = analysis.Analyzer(database_path)

    def onExit(self):
        self.saver.closeDB()

    # This overrides the onNewDepth method in the TraderBase class, so the
    # framework will automatically pick it up and send updates to it.
    def onNewDepth(self, t, pair, asks, bids):
        self.saver.saveDepth(t, pair, asks, bids)

        self.analyzer.analyze(t, pair)

        if pair == "ltc_usd":
            ask_price, ask_amount = asks[0]
            bid_price, bid_amount = bids[0]
            logger.info("LTC/USD Ask: %s, Bid: %s" % (ask_price, bid_price))

    # This overrides the onNewTradeHistory method in the TraderBase class, so the
    # framework will automatically pick it up and send updates to it.
    def onNewTradeHistory(self, t, pair, trades):
        history = self.trade_history_seen.setdefault(pair, set())

        new_trades = filter(lambda trade: trade.tid not in history, trades)
        if new_trades:
            # print "%s Entering %d new %s trades" % (t, len(new_trades), pair)
            self.saver.saveTradeHistory(new_trades)
            history.update(t.tid for t in new_trades)

    def onNewTicker(self, t, pair, ticker):
        self.saver.saveTick(t, pair, ticker)
        logger.info("Received ticker for %s - %s" % (pair, ticker))


def onBotError(msg, tracebackText):
    tstr = time.strftime("%Y/%m/%d %H:%M:%S")
    logger.error("%s - %s\n%s\n%s\n" % (tstr, msg, tracebackText, "-"*80))
    open("logger-bot-error.log", "a").write(
        "%s - %s\n%s\n%s\n" % (tstr, msg, tracebackText, "-"*80))

def run(database_path):
    pairs = ("btc_usd", "btc_rur", "btc_eur", "ltc_btc", "ltc_usd",
             "ltc_rur", "ltc_eur")
    botlogger = TradeBot(pairs, database_path)
    #logger= MarketDataLogger(("btc_usd", "ltc_usd"), database_path)

    # Create a bot and add the logger to it.
    bot = btcebot.Bot()
    bot.addTrader(botlogger)

    # Add an error handler so we can print info about any failures
    bot.addErrorHandler(onBotError)

    # The bot will provide the logger with updated information every
    # 60 seconds.
    bot.setCollectionInterval(60)
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

    args = parser.parse_args()
    run(args.db_path)
