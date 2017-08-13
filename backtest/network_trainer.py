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
from sklearn.metrics import mean_squared_error, f1_score, classification_report, confusion_matrix, \
    precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score
import math
import matplotlib.pyplot as plt
import collections
from sklearn.svm import SVC
import gc
from sklearn.linear_model import SGDClassifier

logger = logging.getLogger(__name__)

# ticker_data = ['btcebtcusd', 'bitstampbtcusd']
ticker_data = ['btcebtcusd']
frame_length = 12
total_features = 20


def find_buy_points(ticker):
    client = MongoClient("mongodb://localhost:27017")
    db = client.bitcoinbot
    ticks = db[ticker + '_1T'].find(
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
        db[ticker + '_points_train'].insert_one(
            {'time': point[0], 'price': price, 'type': 'positive', 'resolution': '1T', 'max_change': point[1],
             'max_change_index': point[2], 'min_change': point[3], 'min_change_index': point[4]})


def prepare_data(ticker):
    print('Loading data...')

    client = MongoClient("mongodb://localhost:27017")
    db = client.bitcoinbot
    date_filter = {'$gte': datetime.datetime(2014, 1, 1), '$lt': datetime.datetime(2017, 3, 1)}
    ticks = db[ticker + '_1T'].find(
        {'time': date_filter}).sort([('time', 1)])
    df = pd.DataFrame.from_records(list(ticks), columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df = df.set_index(['time'])
    df = df.bfill()
    # df = df.dropna(0)
    calculate_indicators(df)

    # df = df.dropna(0)

    point_ticks = db[ticker + '_points_train'].find({'time': date_filter},
                                                    {'time': 1, 'price': 1, 'max_change': 1, 'max_change_index': 1,
                                                     'min_change': 1, 'min_change_index': 1, '_id': 0}).sort(
        [('time', 1)])
    points = list(point_ticks)

    sample = points

    skip_items = 200
    lines_in_shard = 1000
    shards = split_in_shards(df, frame_length, lines_in_shard)

    print('Creating frames...')

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

        point_time = point['time']
        point_frame = find_point_frame(frame_length, point_time, shards)

        max_change = float(point['max_change'] * 100)
        min_change = float(abs(point['min_change']) * 100)
        if max_change >= 0.45:
            buy_point = 1
        else:
            buy_point = 0

        try:
            y[i - skip_items] = buy_point
            x[i - skip_items, :] = np.nan_to_num(np.concatenate((
                point_frame['open_lr'].values,
                point_frame['high_lr'].values,
                point_frame['low_lr'].values,
                point_frame['close_lr'].values,
                point_frame['volume_lr'].values,
                point_frame['close_7_sma_-1_r'].values,
                point_frame['rsi_7'].values,
                point_frame['rsi_14'].values,
                point_frame['macd'].values,
                point_frame['rsi_7_30.0_le_10_c'].values,
                point_frame['rsi_14_30.0_le_10_c'].values,
                point_frame['boll_close'].values,
                point_frame['boll_lb_close'].values,
                point_frame['rsi_21'].values,
                point_frame['rsi_7_xu_rsi_21'].values,
                point_frame['close_xu_boll_lb'].values,
                point_frame['close_xu_boll'].values,
                point_frame['close_xu_close_7_sma'].values,
                point_frame['close_xu_close_14_sma'].values,
                point_frame['close_7_sma_xu_close_21_sma'].values
            )))
        except ValueError:
            print('Da fuq')

    print('Found %s buy points from %s samples' % (np.count_nonzero(y), len(sample)))

    np.save(ticker + '_data_x.npy', x)
    np.save(ticker + '_data_y.npy', y)

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


def find_point_frame(frame_length, point_time, shards):
    for k, v in shards.items():
        if point_time > k:
            continue
        return v.loc[:point_time].tail(frame_length)
    raise ValueError


def split_in_shards(df, frame_length, lines_in_shard):
    shards = np.array_split(df, lines_in_shard)
    mapped_shards = {}
    previous_shard = None
    for i, s in enumerate(shards):
        if i == 0:
            mapped_shards[s.index[-1]] = s
        else:
            mapped_shards[s.index[-1]] = pd.concat([previous_shard.tail(frame_length), s])
        previous_shard = s
    return collections.OrderedDict(sorted(mapped_shards.items()))


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

    # poly = preprocessing.PolynomialFeatures(2)
    # train_x = poly.fit_transform(train_x)

    # print('Training network for alpha=%s' % alpha)
    nn = MLPClassifier(alpha=0.001, tol=0.0001, solver='adam', hidden_layer_sizes=(500, 250),
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


def find_regularization():
    x = np.load('data_x.npy')
    y = np.load('data_y.npy')

    print('Training network...')
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.1)
    # poly = preprocessing.PolynomialFeatures(2)
    # train_x = poly.fit_transform(train_x)

    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01]
    alphas_length = len(alphas)
    iterations = range(0, alphas_length)
    train_errors = np.zeros(alphas_length)
    cv_errors = np.zeros(alphas_length)
    for i, alpha in enumerate(alphas):
        print('Training network for alpha=%s' % alpha)
        # nn = MLPClassifier(alpha=alpha, tol=0.0001, solver='adam', hidden_layer_sizes=(180, 180),
        #                    activation='logistic', verbose=True, max_iter=1000)
        # nn.fit(train_x, train_y)
        nn = SGDClassifier(alpha=alpha)
        nn.fit(x, y)

        train_predict = nn.predict(train_x)
        train_error = f1_score(train_y, train_predict)
        print('Train set error: %s' % train_error)
        train_errors[i] = train_error
        print(classification_report(train_y, train_predict))
        print(confusion_matrix(train_y, train_predict))

        test_predict = nn.predict(test_x)
        print('Test set error: %s' % f1_score(test_y, test_predict))
        print(classification_report(test_y, test_predict))
        print(confusion_matrix(test_y, test_predict))
    # result_2 = clf.predict(x)
    # error = f1_error(y, result)

    # return min_f1, min_parameters
    plt.interactive(True)
    plt.plot(iterations, train_errors)
    plt.plot(iterations, cv_errors)
    plt.show()
    input('Press Enter')


def generate_artificial_data(x, y):
    print('Generating artificial data')

    positive_x = x[y == 1]
    negative_x = x[y == 0]

    print('Found %s positive and %s negative samples' % (len(positive_x), len(negative_x)))

    artificial_n = 5

    input_length = frame_length * total_features
    new_samples = np.empty(shape=(len(positive_x) * artificial_n, input_length))

    for sample_i, sample in enumerate(positive_x):
        for art_i in (0, artificial_n - 1):
            new_sample = np.empty(input_length)
            for i in range(0, total_features - 1):
                new_sample[i] = sample[i] + sample[i] * random.uniform(-0.02, 0.02)
            new_index = (sample_i * artificial_n) + art_i
            new_samples[new_index, :] = new_sample

    print('positive_x shape is %s and %s' % (positive_x.shape[0], positive_x.shape[1]))

    print('new_samples shape is %s and %s' % (new_samples.shape[0], new_samples.shape[1]))
    all_x = np.concatenate((x, new_samples))
    all_y = np.concatenate((np.int64(y), np.ones(len(new_samples), dtype=np.dtype('int64'))))

    bincount = np.bincount(all_y)
    print('Final list has %s positive and %s negative samples' % (bincount[1], bincount[0]))

    return all_x, all_y


def find_incremental_regularization():
    print('Training network...')

    # poly = preprocessing.PolynomialFeatures(2)
    # train_x = poly.fit_transform(train_x)

    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01]
    alphas_length = len(alphas)
    iterations = range(0, alphas_length)
    train_errors = np.zeros(alphas_length)
    test_errors = np.zeros(alphas_length)
    cv_errors = np.zeros(alphas_length)
    for i, alpha in enumerate(alphas):
        print('Training network for alpha=%s' % alpha)
        nn = MLPClassifier(alpha=alpha, tol=0.0001, solver='adam', hidden_layer_sizes=(350,),
                           activation='logistic', verbose=True, max_iter=1000, warm_start=True)
        test_data = {}
        for current_ticker in ticker_data:
            print('Loading data for ticker %s' % current_ticker)
            x_ticker = np.load(current_ticker + '_data_x.npy')
            y_ticker = np.load(current_ticker + '_data_y.npy')

            print('Splitting into training set and test set')
            train_x, test_x, train_y, test_y = train_test_split(x_ticker, y_ticker, test_size=0.2)

            print('Scaling features')
            scaler = preprocessing.StandardScaler()
            train_x = scaler.fit_transform(train_x)
            test_x = scaler.transform(test_x)

            train_x, train_y = generate_artificial_data(train_x, train_y)

            test_data[current_ticker] = (test_x, test_y)

            nn.fit(train_x, train_y)

            train_predict = nn.predict(train_x)
            train_error = f1_score(train_y, train_predict)
            print('Train set error: %s' % train_error)
            train_errors[i] = train_error
            print(classification_report(train_y, train_predict))
            print(confusion_matrix(train_y, train_predict))
            print('Done for ticker %s' % current_ticker)

        for ticker_i, current_ticker in enumerate(ticker_data):
            print('Running predictions on the test set from ticker %s' % current_ticker)
            (test_x, test_y) = test_data[current_ticker]
            test_predict = nn.predict(test_x)
            test_predict_proba = nn.predict_proba(test_x)[:, 1]  # predictions for class=1 only
            print('Test set error: %s' % f1_score(test_y, test_predict))
            print(classification_report(test_y, test_predict))
            print(confusion_matrix(test_y, test_predict))
            print(
                'Test_y length is %s, test_predict_proba length is %s' % ((test_y.shape,), (test_predict_proba.shape,)))
            precision, recall, thresholds = precision_recall_curve(test_y, test_predict_proba)
            print('Saving Precision, recall, threshold arrays...')
            np.save('prc/prc_precision_%s_%s.npy' % (alpha, current_ticker), precision)
            np.save('prc/prc_recall_%s_%s.npy' % (alpha, current_ticker), recall)
            np.save('prc/prc_thresholds_%s_%s.npy' % (alpha, current_ticker), thresholds)

        print('Done for alpha %s' % alpha)

    input('Press Enter')


def calculate_indicators(df):
    df_stock = ss.StockDataFrame.retype(df)
    df_stock['rsi_7']
    df_stock['macd']
    df_stock['close_7_sma_-1_r']
    df_stock['open_lr'] = np.log(df_stock['open'] / df_stock['open_-1_s'])
    df_stock['close_lr'] = np.log(df_stock['close'] / df_stock['close_-1_s'])
    df_stock['high_lr'] = np.log(df_stock['high'] / df_stock['high_-1_s'])
    df_stock['low_lr'] = np.log(df_stock['low'] / df_stock['low_-1_s'])
    df_stock['volume_lr'] = np.log(df_stock['volume'] / df_stock['volume_-1_s'])
    df_stock['rsi_7_30.0_le_10_c']
    df_stock['rsi_14_30.0_le_10_c']
    df_stock['boll']
    df_stock['boll_lb']
    df_stock['boll_close'] = np.log(df_stock['boll'] / df_stock['close'])
    df_stock['boll_lb_close'] = np.log(df_stock['boll_lb'] / df_stock['close'])
    df_stock['rsi_14']
    df_stock['rsi_21']
    df_stock['rsi_7_xu_rsi_21']
    df_stock['close_xu_boll_lb']
    df_stock['close_xu_boll']
    df_stock['close_xu_close_7_sma']
    df_stock['close_xu_close_14_sma']
    df_stock['close_7_sma_xu_close_21_sma']
    #    df_stock['cci_5']
    #    df_stock['cci_14']
    #    df_stock['cci_5_-100.0_le_10_c']
    #    df_stock['cci_14_-100.0_le_10_c']

    df_stock.loc[df_stock['rsi_7'] <= 30, 'rsi_buy'] = 1
    df_stock.loc[df_stock['rsi_7'] > 30, 'rsi_buy'] = 0
    df_stock.loc[df_stock['rsi_7'] >= 70, 'rsi_buy'] = -1

    # df['ma_7'] = df['close'].rolling(window=7).mean().bfill()
    # df['ma_25'] = df['close'].rolling(window=25).mean().bfill()
    #
    # ma_35 = df['close'].rolling(window=35).mean().bfill()
    # std_35 = pd.rolling_std(df['close'], window=35).bfill()
    # df['bb_up'] = ma_35 + (std_35 * 2)
    # df['bb_down'] = ma_35 - (std_35 * 2)
    #
    # df['ma_7_diff'] = df['close'] - df['ma_7']
    # df['ma_25_diff'] = df['close'] - df['ma_25']
    # df['bb_up_diff'] = df['bb_up'] - df['close']
    # df['bb_down_diff'] = df['close'] - df['bb_down']


def import_resampled_data(filename, database_name, period):
    print('Loading CSV file...')
    df = pd.read_csv(filename, header=None, sep=",", names=['time', 'last', 'volume'],
                     parse_dates=True, date_parser=dateparse, index_col=0)
    print('Resampling data...')
    df['last'] = pd.to_numeric(df['last']).bfill()
    df['volume'] = pd.to_numeric(df['volume']).bfill()
    price = df.resample(period, how={'last': 'ohlc'})
    volume = df.resample(period, how={'volume': 'sum'})
    volume.columns = pd.MultiIndex.from_tuples([('volume', 'sum')])
    df = pd.concat([price, volume], axis=1)

    print('Uploading to the database...')
    client = MongoClient("mongodb://localhost:27017")
    db = client.bitcoinbot
    counter = 0

    for i, row in df.iterrows():
        counter = counter + 1
        if counter % 20000 == 0:
            print('Processing tick number %s', counter)
        db[database_name + '_' + period].insert_one(
            {'time': i, 'open': row['last']['open'], 'high': row['last']['high'], 'low': row['last']['low'],
             'close': row['last']['close'], 'volume': row['volume']['sum']})

    print('Done.')


def dateparse(time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))


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


def join_ticker_data():
    x = np.empty((0, total_features * frame_length))
    y = np.empty((0))
    for ticker in ticker_data:
        x_ticker = np.load(ticker + '_data_x.npy')
        y_ticker = np.load(ticker + '_data_y.npy')

        print('Normalizing...')
        x_ticker = preprocessing.scale(x_ticker)

        x = np.concatenate((x, x_ticker))
        y = np.concatenate((y, y_ticker))

    np.save('data_x.npy', x)
    np.save('data_y.npy', y)


def show_prc():
    tickers = ['btcebtcusd', 'bitstampbtcusd']
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01]

    for alpha in alphas:
        plt.clf()
        for ticker in tickers:
            precision = np.load("prc/prc_precision_%s_%s.npy" % (alpha, ticker))
            recall = np.load("prc/prc_recall_%s_%s.npy" % (alpha, ticker))

            # Plot Precision-Recall curve
            plt.plot(recall, precision, lw=2, color='navy',
                     label='Precision-Recall curve')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall')
        plt.legend(loc="lower left")
        plt.savefig('plot_%s.png' % alpha)


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    logger.info("starting event profiler")
    # min_error, min_params = train_network()
    # print('Min error = %s' % min_error)
    # print('Alpha=%s, i1=%s, i2=%s, i3=%s' % (min_params[0], min_params[1], min_params[2], min_params[3]))
    # prepare_data('btcebtcusd')

    #    find_buy_points('bitstampbtcusd')
    # for ticker in ticker_data:
    #    periods = ['1T', '3T', '5T', '15T', '30T', '1h', '2h', '4h', '6h', '12h', '1d', '3d']
    #    for period in periods:
    #        import_resampled_data('../../../data/btcnCNY.csv', 'btcnbtccny', period)
    #    find_buy_points('btcnbtccny')
    #     prepare_data('btcnbtccny')
    #     for ticker in ticker_data:
    #         prepare_data(ticker)
    find_incremental_regularization()
    # repeat_training_network()
    # find_buy_points()
    # periods = ['1T', '3T', '5T', '15T', '30T', '1h', '2h', '4h', '6h', '12h', '1d', '3d']
    # for period in periods:
    #    import_resampled_data('../../../data/bitstampUSD.csv', 'bitstampbtcusd', period)

    # show_prc()
    logger.info("finished event profiler process")
