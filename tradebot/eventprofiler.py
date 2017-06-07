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
        data, cv, test = self.db.retrieve_ticks(self.pair, datetime.datetime(2000, 1, 1), datetime.datetime.now())

        # print(data)

        # data = filter_repeating_ticks(data)

        logger.info("all data loaded")

        logger.info("timestamps merged")

        # self.linear_regression(data)
        y_data = data.copy()
        del y_data['volume']

        y = self.calculate_price_change(y_data)
        x = self.calculate_indicators(y_data)

        y_cv_data = cv.copy()
        del y_cv_data['volume']
        y_cv = self.calculate_price_change(y_cv_data)
        x_cv = self.calculate_indicators(y_cv_data)

        y_test_data = test.copy()
        del y_test_data['volume']
        y_test = self.calculate_price_change(y_test_data)
        x_test = self.calculate_indicators(y_test_data)


        scaler = preprocessing.StandardScaler().fit(x)
        x = scaler.transform(x)
        x_cv = scaler.transform(x_cv)
        x_test = scaler.transform(x_test)


        print('len(x) = %s' % len(x))
        print('len(y) = %s' % len(y))

        print('len(x_cv) = %s' % len(x_cv))
        print('len(y_cv) = %s' % len(y_cv))

        print('len(x_test) = %s' % len(x_test))
        print('len(y_test) = %s' % len(y_test))

        # self.logistic_regression(scaler, x, x_cv, x_test, y, y_cv, y_cv_data, y_data, y_test)
        # self.linear_regression(data)

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
