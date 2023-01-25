"""Seasonal Auto Regressive Integrated Mooving Average, SRIMA is an extension of ARIMA that supports the direct modeling of the seasonal component of the series.
"""
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
import math


class sarima():
    """ This class is implementing the forecasting of the time series using the SARIMA model \n
    
    Parameters: \n
    time_series (pandas series): time series to forecast \n
    target (str): name of target column to forecast \n
    test_size_prct (float): the percentage size of the test set from 0 to 1 \n
    p (int): order of the AR part of the model \n
    d (int): order of the I part of the model \n
    q (int): order of the MA part of the model \n
    P (int): order of the seasonal AR part of the model \n
    D (int): order of the seasonal I part of the model \n
    Q (int): order of the seasonal MA part of the model \n
    m (int): number of time steps for a seasonal period \n
    
    Import and usage: \n
    from src.models.sarima import sarima \n
    sarima_model = sarima(time_series, target, test_size_prct, p, d, q, P, D, Q, m) \n
    forecast, test, model_fit = sarima_model.fit() \n
    """

    def __init__(self, time_series, target, test_size_prct, p, d, q, P, D, Q, m):
        self.time_series = time_series
        self.target = target
        self.test_size_prct = test_size_prct
        # SARIMA parameters
        # p: order of the AR part of the model
        self.p = p
        # d: order of the I part of the model
        self.d = d
        # q: order of the MA part of the model
        self.q = q
        # P: order of the seasonal AR part of the model
        self.P = P
        # D: order of the seasonal I part of the model
        self.D = D
        # Q: order of the seasonal MA part of the model
        self.Q = Q
        # m: number of time steps for a seasonal period
        self.m = m

    def help():
        n_params = 10
        doc = "time_series (pandas series): time series to forecast \n target (str): name of target column to forecast \n test_size_prct (float): the percentage size of the test set from 0 to 1 \n p (int): order of the AR part of the model \n d (int): order of the I part of the model \n q (int): order of the MA part of the model \n P (int): order of the seasonal AR part of the model \n D (int): order of the seasonal I part of the model \n Q (int): order of the seasonal MA part of the model \n m (int): number of time steps for a seasonal period \n"
        return n_params, doc

    def fit(self):
        """
        Fit the SARIMA model to the time series.
        """
        # Split the time series into train and test
        test_size = math.ceil(self.test_size_prct * len(self.time_series))
        train = self.time_series[:-test_size]
        test = self.time_series[-test_size:]

        # Fit the SARIMA model
        model = SARIMAX(train, order=(self.p, self.d, self.q), seasonal_order=(self.P, self.D, self.Q, self.m))
        model_fit = model.fit(disp=False)

        # Make predictions
        predictions = model_fit.forecast(test_size)

        return predictions, test, model_fit

    def plot_results(self, predictions, model_fit):
        """
        Plot the results of the SARIMA model.
        """
        # Plot the results
        plt.plot(self.time_series)
        plt.plot(predictions, color='red')
        plt.show()

        # Print the summary of the model
        print(model_fit.summary())

    def evaluate(self, predictions):
        """
        Evaluate the SARIMA model using the mean absolute error.
        """
        # Calculate the mean absolute error
        mae = mean_absolute_error(self.time_series[-len(predictions):], predictions)

        return mae

    def run(self):
        """
        Run the SARIMA model.
        """
        # Fit the model
        predictions, model_fit = self.fit()

        # Plot the results
        self.plot_results(predictions, model_fit)

        # Evaluate the model
        mae = self.evaluate(predictions)

        return mae, predictions, model_fit

