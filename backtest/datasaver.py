from time import sleep

import requests
import json
from pymongo import MongoClient
import logging
import datasaver_config

xbtusd_url = 'https://api.kraken.com/0/public/OHLC?pair=%s&interval=1'
asset_pairs_url = 'https://api.kraken.com/0/public/AssetPairs'

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger('nxt')
formatter = logging.Formatter(log_format)
handler = logging.FileHandler(datasaver_config.log_path)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)


asset_pairs_response = requests.get(asset_pairs_url)
ap_json = json.loads(asset_pairs_response.text)
pairs = ap_json['result']


client = MongoClient("mongodb://localhost:27017")
db = client['backtest']


for pair, value in pairs.items():
    if ".d" in pair or pair == 'XXLMXXBT':
        continue

    logger.info('processing pair %s' % pair)
    get_olhc_url = xbtusd_url % pair
    response = requests.get(get_olhc_url)
    response_json = json.loads(response.text)

    if 'result' not in response_json:
        logger.error("can't fetch pair %s" % pair)
        logger.error(response_json)
        continue

    ohlc = response_json['result'][pair]

    db_pair = "kraken_%s_1T" % pair.lower()

    last_time = 0
    cursor = db[db_pair].find({}).sort([('time', -1)]).limit(1)
    if cursor.count() > 0:
        last_time = cursor[0]['time']

    inserted_rows = 0
    for row in ohlc:
        row_time, row_open, row_high, row_low, row_close, row_vwap, row_volume, row_count = row

        if row_time <= last_time:
            continue

        data = {
            'time': int(row_time),
            'open': float(row_open),
            'high': float(row_high),
            'low': float(row_low),
            'close': float(row_close),
            'vwap': float(row_vwap),
            'volume': float(row_volume),
            'count': int(row_count)
        }

        db[db_pair].insert(data)

        inserted_rows = inserted_rows + 1

    logger.info('inserted %s rows for ticker %s' % (inserted_rows, pair))






