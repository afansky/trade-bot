import datetime
import indicator
import btcebot
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def filter_repeating_ticks(ticks):
    result = []

    previous = None
    for t in ticks:
        if previous is None:
            previous = t
            continue

        if previous['last'] == t['last']:
            if previous['time'] != t['time']:
                previous['time'] = t['time']
            continue
        else:
            result.append(previous)

        previous = t

    if previous is not None:
        if len(result) == 0 or previous != result[-1]:
            result.append(previous)

    return result


class Analyzer(object):
    def __init__(self):
        self.db = None
        self.indicators = [indicator.bollinger_bands]

    def analyze(self, ticks, pair):
        if len(ticks) < 35:
            logger.debug("not enough data, aborting analysis")
            return

        df = create_data_frame(ticks)
        signals = []
        for func in self.indicators:
            signal = func(df)
            if signal is not None:
                signals.append(signal)

        for signal in signals:
            logger.info("Signal from [%s] detected - %s @ %s - %s" % (signal.source, signal.message, pair, signal.last_price))

        return signals


def create_data_frame(ticks):
    return pd.DataFrame.from_records(ticks, index='time', columns=['time', 'last'])