import requests
import json
from pymongo import MongoClient
import logging
import datasaver_config

xbtusd_url = 'https://api.kraken.com/0/public/OHLC?pair=XBTUSD&interval=1'

log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger('nxt')
formatter = logging.Formatter(log_format)
handler = logging.FileHandler(datasaver_config.log_path)
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
logger.addHandler(handler)


response = requests.get(xbtusd_url)
response_json = json.loads(response.text)
ohlc = response_json['result']['XXBTZUSD']

last_time = 0

client = MongoClient("mongodb://localhost:27017")
db = client['backtest']
cursor = db['kraken_xbtusd_1T'].find({}).sort([('time', -1)]).limit(1)
if cursor.count() > 0:
    last_time = cursor[0]['time']


inserted_rows = 0
for row in ohlc:
    row_time, row_open, row_high, row_low, row_close, row_vwap, row_volume, row_count = row

    if row_time <= last_time:
        continue

    data = {
        'time': row_time,
        'open': row_open,
        'high': row_high,
        'low': row_low,
        'close': row_close,
        'vwap': row_vwap,
        'volume': row_volume,
        'count': row_count
    }

    db['kraken_xbtusd_1T'].insert(data)

    inserted_rows = inserted_rows + 1

print('inserted %s rows' % inserted_rows)