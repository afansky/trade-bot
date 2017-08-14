from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader
import datetime
import pandas as pd
import stockstats as ss
import numpy as np
from pandas_highcharts import core as ph
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import JSONRenderer
from rest_framework.response import Response
import time
from pymongo import MongoClient
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import math

# database_name = 'btcebtcusd'
mongo_db_name = 'backtest'
database_name = 'kraken_xxbtzusd'
points_collection = 'kraken_xxbtzusd'

# Create your views here.

def index(request):
    todays_date = datetime.datetime.now().date()
    index = pd.date_range(todays_date - datetime.timedelta(10), periods=1000, freq='D')

    columns = ['A']
    data = np.array([np.arange(1000)]).T
    df = pd.DataFrame(data, index=index, columns=columns).resample('1M', how='ohlc')

    chart = ph.serialize(df, render_to='my-chart', output_type='json')

    template = loader.get_template('dashboard/index.html')
    context = {'chart': chart}
    return HttpResponse(template.render(context, request))


@api_view(['POST'])
@renderer_classes((JSONRenderer,))
def series(request):
    """
    A view that returns the count of active users in JSON.
    """

    start_int = int(request.POST['start']) / 1000
    start = datetime.datetime.fromtimestamp(start_int)
    end_int = int(request.POST['end']) / 1000
    end = datetime.datetime.fromtimestamp(end_int)

    delta_days = (end - start).days
    delta_hours = -1
    if delta_days <= 1:
        delta_hours = (end - start).seconds // 3600

    client = MongoClient("mongodb://localhost:27017")
    db = client[mongo_db_name]

    if delta_days == 0 and 0 < delta_hours <= 2:
        resolution = '1T'
        resolution_second = 60
    elif delta_days == 0 and 2 < delta_hours <= 6:
        resolution = '3T'
        resolution_second = 60 * 3
    elif delta_days == 0 and 6 < delta_hours <= 12:
        resolution = '5T'
        resolution_second = 60 * 5
    elif delta_days <= 1 and (delta_hours > 12 or (delta_days == 1 and delta_hours >= 0)):
        resolution = '15T'
        resolution_second = 60 * 15
    elif 1 < delta_days <= 3:
        resolution = '30T'
        resolution_second = 60 * 30
    elif 3 < delta_days < 8:
        resolution = '1h'
        resolution_second = 60 * 60
    elif 8 <= delta_days < 16:
        resolution = '2h'
        resolution_second = 60 * 60 * 2
    elif 16 <= delta_days < 30:
        resolution = '4h'
        resolution_second = 60 * 60 * 4
    elif 30 <= delta_days < 60:
        resolution = '6h'
        resolution_second = 60 * 60 * 6
    elif 60 <= delta_days < 120:
        resolution = '12h'
        resolution_second = 60 * 60 * 12
    elif 120 <= delta_days < 240:
        resolution = '1d'
        resolution_second = 60 * 60 * 24
    else:
        resolution = '3d'
        resolution_second = 60 * 60 * 24 * 3


    # for testing
    resolution = '1T'
    resolution_second = 60

    ticks = db[database_name + '_' + resolution].find(
        {'time': {'$lt': end_int, '$gte': start_int - 35 * resolution_second}}).sort([('_id', 1)])

    df = pd.DataFrame.from_records(list(ticks), columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df = df.set_index(['time'])
    df = df.bfill()
    # calculate_indicators(df)
    # df = df[35:]

    data = []
    for i, row in df.iterrows():
        point_time = i * 1000

        point_type = 'unselected'

        data.append(
            [point_time, float(row['open']), float(row['high']), float(row['low']), float(row['close']), float(row['volume']), point_type])

    response = {
        'data': data,
        'resolution': resolution
    }

    return Response(response)


def calculate_indicators(df):
    df_stock = ss.StockDataFrame.retype(df)
    df_stock['rsi_7']
    df_stock['close_7_sma']
    df_stock['close_25_sma']
    df_stock['boll_ub']
    df_stock['boll_lb']

    df['ma_7'] = df['close'].rolling(window=7).mean().bfill()
    df['ma_25'] = df['close'].rolling(window=25).mean().bfill()

    ma_35 = df['close'].rolling(window=35).mean().bfill()
    std_35 = pd.rolling_std(df['close'], window=35).bfill()
    df['bb_up'] = ma_35 + (std_35 * 2)
    df['bb_down'] = ma_35 - (std_35 * 2)


@api_view(['GET'])
@renderer_classes((JSONRenderer,))
def init(request):
    client = MongoClient("mongodb://localhost:27017")
    db = client[mongo_db_name]
    end = db[database_name + "_1T"].find({}, {'time': 1, '_id': 0}).sort([('time', -1)]).limit(1).next()['time']
    start = db[database_name + "_1T"].find({}, {'time': 1, '_id': 0}).sort([('time', 1)]).limit(1).next()['time']
    response = {'start': start * 1000, 'end': end * 1000}
    return Response(response)


@api_view(['GET'])
@renderer_classes((JSONRenderer,))
def points(request):
    tick_list = load_selected_points()
    return Response(tick_list)


def load_selected_points():
    client = MongoClient("mongodb://localhost:27017")
    db = client[mongo_db_name]
    ticks = db[points_collection].find({}, {'time': 1, 'price': 1, 'type': 1, 'resolution': 1, '_id': 0}).sort(
        [('time', -1)])
    tick_list = list(ticks)
    for tick in tick_list:
        tick['time'] = int(tick['time'] * 1000)
    return tick_list


@api_view(['POST'])
@renderer_classes((JSONRenderer,))
def add_point(request):
    point_time = datetime.datetime.fromtimestamp(int(request.POST['time']) / 1000)
    point_resolution = request.POST['resolution']
    client = MongoClient("mongodb://localhost:27017")
    db = client[mongo_db_name]
    tick = db[database_name + '_' + point_resolution].find_one({'time': point_time})

    point_type = request.POST['type']
    success = False
    if tick is not None and not math.isnan(tick['close']):
        db[points_collection].insert_one(
            {'time': point_time, 'price': tick['close'], 'type': point_type, 'resolution': point_resolution})
        success = True

    return Response(success)


@api_view(['POST'])
@renderer_classes((JSONRenderer,))
def remove_point(request):
    point_time = datetime.datetime.fromtimestamp(int(request.POST['time']) / 1000)
    client = MongoClient("mongodb://localhost:27017")
    db = client[mongo_db_name]
    db[points_collection].remove({'time': point_time})
    return Response(True)


def rsi(df):
    rsi_delta = df.diff()[1:]
    delta_up = rsi_delta.copy()
    delta_down = rsi_delta.copy()
    delta_up[delta_up < 0] = 0
    delta_down[delta_down > 0] = 0
    average_gain = pd.rolling_mean(delta_up, 7)
    average_loss = pd.rolling_mean(delta_down.abs(), 7)

    rs_value = average_gain / average_loss
    rsi_result = 100.0 - (100.0 / (1.0 + rs_value))
    rsi_result = rsi_result.bfill()
    return rsi_result