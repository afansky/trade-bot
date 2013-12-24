import btcebot
import datetime


def moving_average(pair, asks, bids):
    db = btcebot.MarketDatabase('btce.sqlite')
    ticks = db.retrieveTicks(pair, datetime.date(2013, 12, 23), datetime.date(2013, 12, 25))
    for tick in ticks:
        print tick
    db.close()