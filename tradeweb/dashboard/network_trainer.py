from pymongo import MongoClient
import datetime
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn import preprocessing
import logging
import stockstats as ss
import random
from sklearn import svm
from sklearn.metrics import mean_squared_error, f1_score
from keras.layers import Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import math

logger = logging.getLogger(__name__)


def find_buy_points():
    client = MongoClient("mongodb://localhost:27017")
    db = client.bitcoinbot
    ticks = db.btcebtcusd_1T.find(
        {'time': {'$gte': datetime.datetime(2013, 1, 1), '$lt': datetime.datetime(2017, 3, 5)}}).sort([('time', 1)])
    df = pd.DataFrame.from_records(list(ticks), columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df = df.set_index(['time'])
    df = df.bfill()
    calculate_indicators(df)

    count = 0
    points = []
    for i, c in enumerate(df.index):
        if i % 10000 == 0:
            print('Processing item #%s' % i)

        if i < 35:
            continue

        if i > len(df.index) - 15:
            break

        price = df.iloc[i]['close']

        max_change = 0
        max_change_index = 0
        min_change = 0
        min_change_index = 0
        for j in (1, 5):

            next_high_price = df.iloc[i + j]['high']
            next_low_price = df.iloc[i + j]['low']
            up_change = math.log(float(next_high_price / price))
            down_change = math.log(float(next_low_price / price))
            if up_change > max_change:
                max_change = up_change
                max_change_index = j

            if down_change < min_change:
                min_change = down_change
                min_change_index = j

        if max_change_index != 0 or min_change_index != 0:
            count = count + 1
            points.append([c, max_change, max_change_index, min_change, min_change_index])

    print('Found %s points' % count)

    for i, point in enumerate(points):
        if i % 10000 == 0:
            print('Importing point #%s' % i)

        price = df.loc[point[0]]['close']
        db['btcebtcusd_points_train'].insert_one(
            {'time': point[0], 'price': price, 'type': 'positive', 'resolution': '1T', 'max_change': point[1],
             'max_change_index': point[2], 'min_change': point[3], 'min_change_index': point[4]})


def prepare_data():
    frame_length = 20

    print('Loading data...')

    client = MongoClient("mongodb://localhost:27017")
    db = client.bitcoinbot
    date_filter = {'$gte': datetime.datetime(2016, 11, 1), '$lt': datetime.datetime(2017, 1, 5)}
    ticks = db.btcebtcusd_1T.find(
        {'time': date_filter}).sort([('time', 1)])
    df = pd.DataFrame.from_records(list(ticks), columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df = df.set_index(['time'])
    df = df.bfill()
    calculate_indicators(df)

    point_ticks = db.btcebtcusd_points_train.find({'time': date_filter},
                                                  {'time': 1, 'price': 1, 'max_change': 1, 'max_change_index': 1,
                                                   'min_change': 1, 'min_change_index': 1, '_id': 0}).sort(
        [('time', 1)])
    points = list(point_ticks)

    sample_size = 3
    sample = points
    # positive_sample = random.sample(points_positive, 5)
    # negative_sample = random.sample(points_negative, 15)

    skip_items = 50

    print('Creating frames...')
    total_features = 23
    total_x = len(sample)
    print('Total amount of frames: %s' % total_x)
    y = np.empty([total_x - skip_items])
    x = np.empty([total_x - skip_items, frame_length * total_features])

    latest_timestamp = datetime.datetime.now()
    for i, point in enumerate(sample):
        if i < skip_items:
            continue

        if i % 1000 == 0:
            seconds_delta = (datetime.datetime.now() - latest_timestamp).seconds
            latest_timestamp = datetime.datetime.now()
            total_seconds_estimate = (total_x / 1000 - i / 1000) * seconds_delta
            print('Processing item #%s, last batch took %s seconds, est. left: %s' % (
                i, seconds_delta, str(datetime.timedelta(seconds=total_seconds_estimate))))

        point_frame = df.loc[:point['time']].tail(frame_length)
        max_change = float(point['max_change'] * 100)
        min_change = float(abs(point['min_change']) * 100)
        if max_change >= 0.31 and min_change <= 0.12:
            buy_point = 1
        else:
            buy_point = 0

        y[i - skip_items] = buy_point
        x[i - skip_items, :] = np.nan_to_num(np.concatenate((
            point_frame['open_-1_r'].values,
            point_frame['high_-1_r'].values,
            point_frame['low_-1_r'].values,
            point_frame['close_-1_r'].values,
            point_frame['volume_-1_r'].values,
            point_frame['ma_7'].values,
            point_frame['ma_25'].values,
            point_frame['bb_up'].values,
            point_frame['bb_down'].values,
            point_frame['ma_7_diff'].values,
            point_frame['ma_25_diff'].values,
            point_frame['rsi_7'].values,
            point_frame['rsi_25'].values,
            point_frame['macd'].values,
            point_frame['macds'].values,
            point_frame['macdh'].values,
            point_frame['kdjk'].values,
            point_frame['kdjd'].values,
            point_frame['kdjj'].values,
            point_frame['pdi'].values,
            point_frame['mdi'].values,
            point_frame['bb_up_diff'].values,
            point_frame['bb_down_diff'].values
        )))

    print('Normalizing...')
    x = preprocessing.scale(x)

    np.save('network_data_x.npy', x)
    np.save('network_data_y.npy', y)

    # poly = preprocessing.PolynomialFeatures(2)
    # x = poly.fit_transform(x)

    # alphas = np.logspace(-5, 3, 5)
    # min_f1 = 1
    # min_parameters = []
    # for alpha in alphas:
    #     for i1 in range(100, 150):
    #         for i2 in range(100, 150):
    #             for i3 in range(100, 150):
    #                 nn = MLPClassifier(alpha=alpha,
    #                                    hidden_layer_sizes=(i1, i2, i3), activation='logistic')
    #                 nn.fit(x, y)
    #                 result = nn.predict(x)
    #                 error = f1_error(y, result)
    #                 if error < min_f1:
    #                     print('Found min error')
    #                     min_f1 = error
    #                     min_parameters = [alpha, i1, i2, i3]
    #                 print("alpha = %s, i1=%s, i2=%s, i3=%s, F1 error = %s" % (alpha, i1, i2, i3, error))

    # print('Training network...')

    # clf = svm.SVC()
    # clf.fit(x,y)

    # nn = MLPClassifier(alpha=0.0001, tol=0.0001, solver='adam', hidden_layer_sizes=(250, 250, 250),
    #                    activation='logistic', verbose=True)
    # nn.fit(x, y)

    # result = nn.predict(x)
    # result_2 = clf.predict(x)
    # error = f1_error(y, result)

    # print('Train set error: %s' % f1_score(y, nn.predict(x)))

    # return min_f1, min_parameters


def repeat_training_network():
    x = np.load('network_data_x.npy')
    y = np.load('network_data_y.npy')

    print('Training network...')

    total_length = len(x)
    cv_data_index = int(total_length * 0.8)
    test_data_index = int(total_length * 0.9)

    train_x = x[:cv_data_index]
    cv_x = x[cv_data_index:test_data_index]
    test_x = x[test_data_index:]

    train_y = y[:cv_data_index]
    cv_y = y[cv_data_index:test_data_index]
    test_y = y[test_data_index:]

    # print('Training network for alpha=%s' % alpha)
    nn = MLPClassifier(alpha=0.00001, tol=0.0001, solver='adam', hidden_layer_sizes=(1000, 250),
                       activation='logistic', verbose=True)
    nn.fit(train_x, train_y)

    # clf = svm.SVC()
    # clf.fit(x,y)

    print('Train set error: %s' % f1_score(train_y, nn.predict(train_x)))
    print('CV set error: %s' % f1_score(cv_y, nn.predict(cv_x)))
    print('Test set error: %s' % f1_score(test_y, nn.predict(test_x)))
    # result_2 = clf.predict(x)
    # error = f1_error(y, result)

    # return min_f1, min_parameters


def repeat_training_network_keras():
    x = np.load('network_data_x.npy')
    y = np.load('network_data_y.npy')

    y = y * 1000

    print('Training network...')

    total_length = len(x)
    total_features = len(x[0])
    cv_data_index = int(total_length * 0.8)
    test_data_index = int(total_length * 0.9)

    train_x = x[:cv_data_index]
    cv_x = x[cv_data_index:test_data_index]
    test_x = x[test_data_index:]

    train_y = y[:cv_data_index]
    cv_y = y[cv_data_index:test_data_index]
    test_y = y[test_data_index:]

    # print('Training network with layer=%s' % layer)
    model = Sequential()
    model.add(LSTM(250, input_dim=total_features))
    model.add(LSTM(250, input_dim=total_features))
    model.add(LSTM(250, input_dim=total_features))
    model.add(Activation('relu'))

    nn = MLPRegressor(alpha=0.0001, tol=0.0001, solver='adam', hidden_layer_sizes=(1000, 250),
                      activation='tanh', verbose=True)
    nn.fit(train_x, train_y)

    # clf = svm.SVC()
    # clf.fit(x,y)

    print('Train set error: %s' % mean_squared_error(train_y, nn.predict(train_x)))
    print('CV set error: %s' % mean_squared_error(cv_y, nn.predict(cv_x)))
    print('Test set error: %s' % mean_squared_error(test_y, nn.predict(test_x)))
    # result_2 = clf.predict(x)
    # error = f1_error(y, result)

    # return min_f1, min_parameters


def calculate_indicators(df):
    df_stock = ss.StockDataFrame.retype(df)
    df_stock['rsi_7']
    df_stock['rsi_25']
    df_stock['macd']
    df_stock['macds']
    df_stock['macdh']
    df_stock['kdjk']
    df_stock['kdjd']
    df_stock['kdjj']
    df_stock['pdi']
    df_stock['mdi']
    df_stock['close_7_sma']
    df_stock['close_25_sma']
    df_stock['boll_ub']
    df_stock['boll_lb']
    df_stock['open_-1_r']
    df_stock['close_-1_r']
    df_stock['high_-1_r']
    df_stock['low_-1_r']
    df_stock['volume_-1_r']

    df['ma_7'] = df['close'].rolling(window=7).mean().bfill()
    df['ma_25'] = df['close'].rolling(window=25).mean().bfill()

    ma_35 = df['close'].rolling(window=35).mean().bfill()
    std_35 = pd.rolling_std(df['close'], window=35).bfill()
    df['bb_up'] = ma_35 + (std_35 * 2)
    df['bb_down'] = ma_35 - (std_35 * 2)

    df['ma_7_diff'] = df['close'] - df['ma_7']
    df['ma_25_diff'] = df['close'] - df['ma_25']
    df['bb_up_diff'] = df['bb_up'] - df['close']
    df['bb_down_diff'] = df['close'] - df['bb_down']


def f1_error(y, z):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for j, p in enumerate(z):
        if y[j] == 1:
            if p == 1:
                true_positive = true_positive + 1
            else:
                false_positive = false_positive + 1
        else:
            if p == 1:
                false_negative = false_negative + 1
    if true_positive + false_positive == 0 or true_positive + false_negative == 0:
        return 1
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    error = (2 * precision * recall) / (precision + recall)
    return error


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    logger.info("starting event profiler")
    # min_error, min_params = train_network()
    # print('Min error = %s' % min_error)
    # print('Alpha=%s, i1=%s, i2=%s, i3=%s' % (min_params[0], min_params[1], min_params[2], min_params[3]))

    prepare_data()
    # repeat_training_network()
    # find_buy_points()

    logger.info("finished event profiler process")
