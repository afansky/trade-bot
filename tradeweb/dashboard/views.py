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
def series(request, format=None):
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
        response.append([int(time.mktime(i.timetuple())) * 1000, row['last']['open'], row['last']['high'], row['last']['low'], row['last']['close'], row['volume']['sum']])

    return Response(response)
