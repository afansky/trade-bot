from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader
import datetime
import pandas as pd
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


@api_view(['GET'])
@renderer_classes((JSONRenderer,))
def series_old(request, format=None):
    """
    A view that returns the count of active users in JSON.
    """

    client = MongoClient("mongodb://localhost:27017")
    db = client.bitcoinbot
    ticks = db.btcebtcusd.find({}).sort([('_id', -1)]).limit(50000)

    df = pd.DataFrame.from_records(list(ticks), columns=['time', 'last', 'volume'])
    df = df.set_index(['time'])

    price = df.resample('10T', how={'last': 'ohlc'})
    volume = df.resample('10T', how={'volume': 'sum'})
    volume.columns = pd.MultiIndex.from_tuples([('volume', 'sum')])
    df = pd.concat([price, volume], axis=1)
    # for i, row in df.iterrows():
    #     response.append([int(time.mktime(i.timetuple())) * 1000, row['last'], row['volume']])

    # df = pd.DataFrame(list(ticks), index=['time'], columns=['last', 'volume']).resample('1T', how='ohlc')

    response = []
    for i, row in df.iterrows():
        response.append(
            [int(time.mktime(i.timetuple())) * 1000, row['last']['open'], row['last']['high'], row['last']['low'],
             row['last']['close'], row['volume']['sum']])

    return Response(response)


@api_view(['POST'])
@renderer_classes((JSONRenderer,))
def series(request):
    """
    A view that returns the count of active users in JSON.
    """

    start = datetime.datetime.fromtimestamp(int(request.POST['start']) / 1000)
    end = datetime.datetime.fromtimestamp(int(request.POST['end']) / 1000)

    delta_days = (end - start).days
    delta_hours = -1
    if delta_days <= 1:
        delta_hours = (end - start).seconds // 3600

    client = MongoClient("mongodb://localhost:27017")
    db = client.bitcoinbot

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

    ticks = db['btcebtcusd_' + resolution].find(
        {'time': {'$lt': end, '$gte': start - datetime.timedelta(seconds=35 * resolution_second)}}).sort([('_id', 1)])

    df = pd.DataFrame.from_records(list(ticks), columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df = df.set_index(['time'])
    df = df.bfill()
    calculate_indicators(df)
    df = df[35:]

    positive_points = load_selected_points()

    data = []
    for i, row in df.iterrows():
        point_time = int(time.mktime(i.timetuple())) * 1000

        point_type = 'unselected'
        for p in positive_points:
            if p['time'] == point_time:
                point_type = p['type']

        data.append(
            [point_time, row['open'], row['high'], row['low'], row['close'], row['volume'],
             row['ma_7'], row['ma_25'], row['rsi'], row['bb_up'], row['bb_down'], point_type])

    response = {
        'data': data,
        'resolution': resolution
    }

    return Response(response)


def train_network():
    client = MongoClient("mongodb://localhost:27017")
    db = client.bitcoinbot
    ticks = db.btcebtcusd_2h.find({}).sort([('_id', 1)])
    df = pd.DataFrame.from_records(list(ticks), columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df = df.set_index(['time'])
    df = df.bfill()
    calculate_indicators(df)

    point_ticks = db.btcebtcusd_points.find({}, {'time': 1, 'price': 1, '_id': 0})
    point_tick_list = list(point_ticks)

    frames = []
    for point in point_tick_list:
        point_frame = df.loc[:point['time']].tail(20)
        frames.append(point_frame)

    total_x = len(frames)
    frame_length = 20
    total_features = len(df.columns)

    x = np.empty([total_x, frame_length * total_features])
    y = np.empty(total_x)
    for i, c in enumerate(frames):
        y[i] = 1
        x[i, :] = np.nan_to_num(np.concatenate((
            c['open'].values,
            c['high'].values,
            c['low'].values,
            c['close'].values,
            c['volume'].values,
            c['ma_7'].values,
            c['ma_25'].values,
            c['rsi'].values,
            c['bb_up'].values,
            c['bb_down'].values
        )))

    x = preprocessing.normalize(x)
    nn = MLPClassifier(alpha=1e-5,
                           hidden_layer_sizes=(176, 176, 176, 176), activation='logistic')

    nn.fit(x, y)

    result = nn.predict(x)
    return result


def calculate_indicators(df):
    df['ma_7'] = df['close'].rolling(window=7).mean().bfill()
    df['ma_25'] = df['close'].rolling(window=25).mean().bfill()
    df['rsi'] = rsi(df['close'])
    ma_35 = df['close'].rolling(window=35).mean().bfill()
    std_35 = pd.rolling_std(df['close'], window=35).bfill()
    df['bb_up'] = ma_35 + (std_35 * 2)
    df['bb_down'] = ma_35 - (std_35 * 2)


@api_view(['GET'])
@renderer_classes((JSONRenderer,))
def init(request):
    client = MongoClient("mongodb://localhost:27017")
    db = client.bitcoinbot
    end = db.btcebtcusd_2h.find({}, {'time': 1, '_id': 0}).sort([('time', -1)]).limit(1).next()['time']
    start = db.btcebtcusd_2h.find({}, {'time': 1, '_id': 0}).sort([('time', 1)]).limit(1).next()['time']
    response = {'start': time.mktime(start.timetuple()) * 1000, 'end': time.mktime(end.timetuple()) * 1000}
    return Response(response)


@api_view(['GET'])
@renderer_classes((JSONRenderer,))
def points(request):
    tick_list = load_selected_points()
    return Response(tick_list)


def load_selected_points():
    client = MongoClient("mongodb://localhost:27017")
    db = client.bitcoinbot
    ticks = db.btcebtcusd_points.find({}, {'time': 1, 'price': 1, 'type': 1, 'resolution': 1, '_id': 0})
    tick_list = list(ticks)
    for tick in tick_list:
        tick['time'] = int(time.mktime(tick['time'].timetuple()) * 1000)
    return tick_list


@api_view(['POST'])
@renderer_classes((JSONRenderer,))
def add_point(request):
    point_time = datetime.datetime.fromtimestamp(int(request.POST['time']) / 1000)
    point_resolution = request.POST['resolution']
    client = MongoClient("mongodb://localhost:27017")
    db = client.bitcoinbot
    tick = db['btcebtcusd_' + point_resolution].find_one({'time': point_time})

    point_type = request.POST['type']
    success = False
    if tick is not None and not math.isnan(tick['close']):
        db.btcebtcusd_points.insert_one(
            {'time': point_time, 'price': tick['close'], 'type': point_type, 'resolution': point_resolution})
        success = True

    return Response(success)


@api_view(['POST'])
@renderer_classes((JSONRenderer,))
def remove_point(request):
    point_time = datetime.datetime.fromtimestamp(int(request.POST['time']) / 1000)
    client = MongoClient("mongodb://localhost:27017")
    db = client.bitcoinbot
    db.btcebtcusd_points.remove({'time': point_time})
    return Response(True)


def rsi(df):
    rsi_delta = df.diff().bfill()
    delta_up = rsi_delta.copy()
    delta_down = rsi_delta.copy()
    delta_up[delta_up < 0] = 0
    delta_down[delta_down > 0] = 0
    average_gain = pd.rolling_mean(delta_up, 3).bfill()
    average_loss = pd.rolling_mean(delta_down, 3).bfill().abs()

    rs_value = average_gain / average_loss
    rs_value = rs_value.fillna(method='bfill')
    rsi_result = 100.0 - (100.0 / (1.0 + rs_value))
    rsi_result = rsi_result.bfill()
    return rsi_result


@api_view(['GET'])
@renderer_classes((JSONRenderer,))
def train(request):
    return train_network()
