import logging
from analysis.portfolio import Portfolio
import database
from analysis.analysis import filter_repeating_ticks, Analyzer
from analysis import indicator
import datetime
from simulator import merge_timestamps
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import svm
from scipy.signal import argrelextrema
import mlpy
from sklearn.neural_network import MLPClassifier
import time
import matplotlib.pyplot as plt
import os.path
from pymongo import MongoClient

logger = logging.getLogger(__name__)


class EventProfiler:
    def __init__(self, trade_pair, initial_portfolio=None):
        self.pair = trade_pair
        self.db = database.CsvDatabase()
        self.analyzer = Analyzer()
        self.portfolio = Portfolio(initial_portfolio)
        self.signal_count = 0

    def profile(self):
        logger.info("loading data for %s" % self.pair)
        self.db.import_resampled_data('1H')


    def test_network(self):
        nn = self.train_network()
        data = self.db.load_all_data()
        figure_length = 88
        data_length = len(data.index)
        predictions = np.zeros(data_length)
        frames = np.empty(data_length, dtype=pd.DataFrame)
        for i in range(1, data_length - figure_length, 22):
            subset = data[i:i + figure_length]
            subset_norm = subset.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
            frames[i] = subset_norm
            # subset_norm.plot()
            x = np.nan_to_num(np.concatenate((subset_norm['last'].values, subset_norm['volume'].values))).reshape(1, -1)
            response = nn.predict(x)[0]
            if int(response) == 1:
                predictions[i] = data['last'][i]
            print('Frame %s has been predicted as %s with the sample' % (i, predictions[i]))
        data['predictions'] = predictions
        plt.interactive(True)
        data.plot(style=['-', '-', '.'])
        plt.show()
        input('Press Enter')

    def train_network(self):
        min_errors = []
        for i in range(1, 300):
            filename = "/Users/afansky/dev/code/bitcoin_bot/data/hs_down/%s.csv" % i
            if os.path.isfile(filename):
                df = pd.DataFrame.from_csv(filename)
                min_errors.append(df)
        print('loaded %s samples' % len(min_errors))
        max_errors = []
        for i in range(1, 5000):
            filename = "/Users/afansky/dev/code/bitcoin_bot/data/hs_down_negative/%s.csv" % i
            if os.path.isfile(filename):
                df = pd.DataFrame.from_csv(filename)
                max_errors.append(df)
        print('loaded %s samples' % len(max_errors))
        figure_length = len(min_errors[0].index)
        all_errors = min_errors + max_errors
        total_x = len(min_errors) + len(max_errors)
        x = np.empty([total_x, figure_length * 2])
        y = np.empty(total_x)
        for i, c in enumerate(all_errors):
            if i < len(min_errors):
                y[i] = 1
            else:
                y[i] = 0

            x[i, :] = np.nan_to_num(np.concatenate((c['last'].values, c['volume'].values)))
        nn = MLPClassifier(alpha=1e-5,
                           hidden_layer_sizes=(176, 176, 176, 176), activation='logistic')
        nn.fit(x, y)
        # for i, c in enumerate(x):
        #     prediction = nn.predict(x[i].reshape(1, -1))
        #     print('prediction for item #%s = %s' % (i, prediction))

        return nn

    def find_unsimilar_samples(self):
        data = self.db.load_all_data()
        df = self.db.load_samples()
        df_norm = df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
        figure_length = len(df_norm.index)
        data_length = len(data.index)
        errors = np.zeros(data_length)
        inverted_errors = np.zeros(data_length)
        inverted_errors_large = np.zeros(data_length)
        frames = np.empty(data_length, dtype=pd.DataFrame)
        for i, c in enumerate(data.index):
            if i >= data_length - figure_length:
                break

            subset = data[i:i + figure_length]
            subset_norm = subset.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
            frames[i] = subset_norm
            # subset_norm.plot()
            distance = mlpy.dtw_std(subset_norm['last'].values, df_norm['last'].values, dist_only=True)
            if distance == np.Inf or distance == np.NaN:
                distance = 100
            elif distance < 7.5:
                inverted_errors[i] = 400
            elif distance > 20.0:
                inverted_errors_large[i] = 800
            errors[i] = distance
            print('Frame %s has distance %s with the sample' % (i, distance))

        max_errors = []
        for i, c in enumerate(errors):
            if c > 20.0 and c < 100.0:
                max_errors.append(frames[i])

        for i, frame in enumerate(max_errors):
            frame.to_csv('/Users/afansky/dev/code/bitcoin_bot/data/hs_down_negative/%s.csv' % i)


    def find_similar_samples(self):
        data = self.db.load_all_data()
        df = self.db.load_samples()
        df_norm = df.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
        figure_length = len(df_norm.index)
        data_length = len(data.index)
        errors = np.zeros(data_length)
        inverted_errors = np.zeros(data_length)
        inverted_errors_large = np.zeros(data_length)
        frames = np.empty(data_length, dtype=pd.DataFrame)
        for i, c in enumerate(data.index):
            if i >= data_length - figure_length:
                break

            subset = data[i:i + figure_length]
            subset_norm = subset.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
            frames[i] = subset_norm
            # subset_norm.plot()
            distance = mlpy.dtw_std(subset_norm['last'].values, df_norm['last'].values, dist_only=True)
            if distance == np.Inf or distance == np.NaN:
                distance = 100
            elif distance < 7.5:
                inverted_errors[i] = 400
            elif distance > 20.0:
                inverted_errors_large[i] = 800
            errors[i] = distance
            print('Frame %s has distance %s with the sample' % (i, distance))
        data['errors'] = inverted_errors
        data['large_errors'] = inverted_errors_large
        local_min_errors = errors[argrelextrema(errors, np.less, order=30)]
        local_min_frames = frames[argrelextrema(errors, np.less, order=30)]
        valid_samples_size = sum(i < 10.0 for i in local_min_errors)
        valid_frames = []
        valid_errors = []
        for i, c in enumerate(local_min_errors):
            if c < 10.0:
                valid_frames.append(local_min_frames[i])
                valid_errors.append(c)
        min_errors_responses = np.zeros(len(valid_errors))
        plt.interactive(True)
        total_length = len(valid_frames)
        for i, c in enumerate(valid_frames):
            print("Processing item number %s with error %s from %s (%.2f%%)" % (
            i, valid_errors[i], total_length, float((i / total_length) * 100)))
            valid_frames[i].plot()
            plt.show()
            response = input("Does this look like a correct sample?\r\n")
            if response == '1':
                min_errors_responses[i] = 1
            else:
                min_errors_responses[i] = 0
        for i, response in enumerate(min_errors_responses):
            if response == 1:
                valid_frames[i].to_csv('/Users/afansky/dev/code/bitcoin_bot/data/hs_down/%s.csv' % i)

    def train_neural_network(self, errors, frames, figure_length):
        min_distance = np.amin(errors)
        avg_distance = np.average(errors)
        min_errors = frames[np.where(errors < 5.0)]
        max_errors = frames[np.where(errors > 20.0)]

        all_errors = np.concatenate((min_errors, max_errors))
        total_x = len(min_errors) + len(max_errors)
        x = np.empty([total_x, figure_length * 2])
        y = np.empty(total_x)
        for i, c in enumerate(all_errors):
            if i < len(min_errors):
                y[i] = 1
            else:
                y[i] = 0

            x[i, :] = np.nan_to_num(np.concatenate((c['last'].values, c['volume'].values)))
        nn = MLPClassifier(solver='lbfgs', alpha=1e-5,
                           hidden_layer_sizes=(50, 2), activation='logistic')
        nn.fit(x, y)
        for i, c in enumerate(x):
            prediction = nn.predict(x[i].reshape(1, -1))
            print('prediction for item #%s = %s' % (i, prediction))

    def find_trends(self, learn):
        local_min = argrelextrema(learn['last'].values, np.less, order=300)
        local_min_values = learn['last'].values[local_min]
        new_min = []
        for i, c in enumerate(learn.index):
            if i in local_min[0]:
                new_min.append(learn['last'][c])
            else:
                new_min.append(0)
        learn['min'] = new_min
        linreg = linear_model.LinearRegression(normalize=True)
        linreg.fit(local_min[0].reshape(2, 1), local_min_values.reshape(2, 1))
        print('Coefficients: \n', linreg.coef_)
        regression = []
        for i, c in enumerate(learn.index):
            regression.append(linreg.predict(i)[0][0])
        learn['min_regression'] = regression
        local_max = argrelextrema(learn['last'].values, np.greater, order=500)
        local_max_values = learn['last'].values[local_max]
        new_max = []
        for i, c in enumerate(learn.index):
            if i in local_max[0]:
                new_max.append(learn['last'][c])
            else:
                new_max.append(0)
        learn['max'] = new_max
        linreg = linear_model.LinearRegression(normalize=True)
        linreg.fit(local_max[0].reshape(2, 1), local_max_values.reshape(2, 1))
        print('Coefficients: \n', linreg.coef_)
        regression = []
        for i, c in enumerate(learn.index):
            regression.append(linreg.predict(i)[0][0])
        learn['max_regression'] = regression

        learn.plot()

    def logistic_regression(self, scaler, x, x_cv, x_test, y, y_cv, y_cv_data, y_data, y_test):
        c_list = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000]
        errors = np.zeros(len(c_list))
        for i, c in enumerate(c_list):
            # logreg = svm.SVC(C=c)
            # logreg.fit(x, y)

            logreg = linear_model.LogisticRegression(C=c)
            logreg.fit(x, y)

            z = logreg.predict(x_cv)
            z = z.reshape(len(z), 1)

            print('calculating for c=%s' % c)

            error = self.f1_error(y_cv, z)
            print(error)
            errors[i] = error

            z = logreg.predict(x)
            z = z.reshape(len(z), 1)

            error = self.f1_error(y, z)
            print(error)

            z = logreg.predict(x_test)
            z = z.reshape(len(z), 1)

            error = self.f1_error(y_test, z)
            print(error)
        argmin = errors.argmax()
        print('min error is %s, best c is %s' % (errors[argmin], c_list[argmin]))
        c = c_list[argmin]
        logreg = linear_model.LogisticRegression(C=c_list[argmin])
        logreg.fit(x, y)
        # logreg = svm.SVC(C=c)
        # logreg.fit(x, y)
        z = logreg.predict(x_test)
        z = z.reshape(len(z), 1)
        error = self.f1_error(y_test, z)
        print('regularization error is %s' % error)
        eval_errors = np.zeros(y_data.size - 100)
        cv_errors = np.zeros(y_data.size - 100)
        for i in range(100, 5000):
            print('processing for %s items out of total %s' % (i, y_data.size))
            y_head = y_data.head(i)
            y = self.calculate_price_change(y_head)
            y_cv = self.calculate_price_change(y_cv_data.head(i))
            x = self.calculate_indicators(y_head)
            x_cv = self.calculate_indicators(y_cv_data.head(i))
            x = scaler.transform(x)
            x_cv = scaler.transform(x_cv)

            if max(y) == 0:
                eval_errors[i] = 0
                continue

            logreg = linear_model.LogisticRegression(C=c_list[argmin])
            logreg.fit(x, y)
            # logreg = svm.SVC(C=c)
            # logreg.fit(x, y)

            z = logreg.predict(x)
            z = z.reshape(len(z), 1)

            z_cv = logreg.predict(x_cv)
            z_cv = z_cv.reshape(len(z_cv), 1)

            error = self.f1_error(y, z)
            cv_error = self.f1_error(y_cv, z_cv)
            eval_errors[i] = error
            cv_errors[i] = cv_error
        plt.plot(eval_errors, eval_errors, 'r--', cv_errors, cv_errors, 'bs')
        plt.show()

    def f1_error(self, y, z):
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
        if (true_positive + false_positive == 0 or true_positive + false_negative == 0):
            return 1
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        error = (2 * precision * recall) / (precision + recall)
        return error

    def calculate_price_change(self, y_data):
        pct_change = y_data.pct_change(periods=10, fill_method='pad')
        y = y_data.copy()
        y[pct_change >= 0.03] = 1
        y[pct_change < 0.03] = 0
        y = y[25:]
        y = y[45:].as_matrix()
        return y

    def calculate_indicators(self, y_data):
        data_rsi = indicator.rsi(y_data)[70:]
        data_macd = indicator.macd(y_data)[70:]
        x = np.nan_to_num(pd.concat([data_rsi, data_macd], axis=1).as_matrix())
        return x

    def linear_regression(self, data):
        y_data = data.copy()
        del y_data['volume']
        pct_change = y_data.pct_change(periods=24, fill_method='pad')
        y = y_data.copy()
        y[pct_change >= 0.05] = 1
        y[pct_change < 0.05] = 0
        y = y_data[25:]
        x = y.index.values.reshape(y.size, 1).astype(np.int64)
        y = y.as_matrix()
        # Create linear regression object
        regr = linear_model.LinearRegression()
        # Train the model using the training sets
        regr.fit(x, y)
        # The coefficients
        print('Coefficients: \n', regr.coef_)
        # The mean squared error
        print("Mean squared error: %.2f"
              % np.mean((regr.predict(x) - y) ** 2))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % regr.score(x, y))
        # Plot outputs
        plt.scatter(x, y, color='black')
        plt.plot(x, regr.predict(x), color='blue',
                 linewidth=3)
        plt.xticks(())
        plt.yticks(())
        plt.show()


if __name__ == '__main__':
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    logger.info("starting event profiler")
    start_date = datetime.datetime.now()

    # pairs = ("btc_usd", "btc_rur", "btc_eur", "ltc_btc", "ltc_usd", "ltc_rur", "ltc_eur")
    pair = 'btc_usd'
    profiler = EventProfiler(pair)

    profiler.profile()

    end_date = datetime.datetime.now()
    logger.info("finished event profiler process, it took %s seconds" % (end_date - start_date).seconds)
