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

    client = MongoClient("mongodb://localhost:27017")
    db = client.bitcoinbot
    ticks = db.btcebtcusd_1h.find({'time': {'$lt': end, '$gte': start}}).sort([('_id', 1)])

    df = pd.DataFrame.from_records(list(ticks), columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df = df.set_index(['time'])
    df = df.bfill()

    response = []
    for i, row in df.iterrows():
        response.append(
            [int(time.mktime(i.timetuple())) * 1000, row['open'], row['high'], row['low'], row['close'], row['volume']])

    return Response(response)


@api_view(['GET'])
@renderer_classes((JSONRenderer,))
def init(request):
    start = datetime.datetime.now() - datetime.timedelta(days=45)
    end = datetime.datetime.now()
    response = {'start': time.mktime(start.timetuple()) * 1000, 'end': time.mktime(end.timetuple()) * 1000}
    return Response(response)


@api_view(['GET'])
@renderer_classes((JSONRenderer,))
def points(request):
    client = MongoClient("mongodb://localhost:27017")
    db = client.bitcoinbot
    ticks = db.btcebtcusd_1h_points.find({}, {'time': 1, 'price': 1, '_id': 0})
    tick_list = list(ticks)
    for tick in tick_list:
        tick['time'] = time.mktime(tick['time'].timetuple()) * 1000
    return Response(tick_list)


@api_view(['POST'])
@renderer_classes((JSONRenderer,))
def add_point(request):
    point_time = datetime.datetime.fromtimestamp(int(request.POST['time']) / 1000)
    client = MongoClient("mongodb://localhost:27017")
    db = client.bitcoinbot
    tick = db.btcebtcusd_1h.find_one({'time': point_time})
    db.btcebtcusd_1h_points.insert_one({'time': point_time, 'price': tick['close']})
    return Response(True)


@api_view(['POST'])
@renderer_classes((JSONRenderer,))
def remove_point(request):
    point_time = datetime.datetime.fromtimestamp(int(request.POST['time']) / 1000)
    client = MongoClient("mongodb://localhost:27017")
    db = client.bitcoinbot
    db.btcebtcusd_1h_points.remove({'time': point_time})
    return Response(True)
