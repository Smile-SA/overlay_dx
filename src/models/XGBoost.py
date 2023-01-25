"""XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework.
"""
# Import libraries
import numpy as np
import pandas as pd
import xgboost as xgb
import math
import matplotlib.pyplot as plt
import pandas as pd

class xgboost():
    """class to create the xgboost model \n
    
    Attributes: \n
        df (pandas dataframe): pandas dataframe with the time series as columns \n
        target (str): name of target column to forecast \n
        test_size (float): the percentage size of the test set from 0 to 1 \n

    Import and usage: \n
        from src.models.xgboost import xgboost \n
        xgboost_model = xgboost(df, target, test_size) \n
        pred, test, train = xgboost_model.fit() \n
    """

    def __init__(self, df, target, test_size):
        self.target = target
        self.data = df[target]
        self.test_size = math.floor(test_size*self.data.shape[0])
        print(self.test_size)
        self.train = df[target][:-self.test_size]
        self.test = self.data[-self.test_size:]

    def help():
        n_params = 3
        doc = "df_path (path): path of the pandas dataframe with the time series as columns \n target (str): name of target column to forecast \n test_size (float): the percentage size of the test set from 0 to 1 \n"
        return n_params, doc


    def fit(self):
        data = self.series_to_supervised(self.data)
        forecast = self.walk_forward_validation(data = data,n_test=self.test_size)
        forecast = pd.Series(forecast)
        forecast.index = self.test.index
        self.plot_predictions(forecast)
        return forecast, self.test, self.train


    # transform a time series dataset into a supervised learning dataset
    def series_to_supervised(self,df, n_in=1, n_out=1, dropnan=True):
        cols = list()
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
        for i in range(0, n_out):
            cols.append(df.shift(-i))
        agg = pd.concat(cols, axis=1)
        if dropnan:
            agg.dropna(inplace=True)

        return agg.values
    
    # split a dataset into train/test sets
    def train_test_split(self,data, n_test):
        return data[:-n_test, :], data[-n_test:, :]

    # fit an xgboost model and make a one step prediction
    def xgboost_forecast(self,train, testX):
        # transform list into array
        train = np.asarray(train)
        # split into input and output columns
        trainX, trainy = train[:, :-1], train[:, -1]
        # fit model
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        model.fit(trainX, trainy)
        # make a one-step prediction
        pred = model.predict([testX])
        return pred[0]

    def walk_forward_validation(self,data, n_test):
        print("ntest",n_test)
        forecast = list()
        # split dataset
        train, test = self.train_test_split(data,n_test)
        # step over each time-step in the test set
        for i in range(len(test)):
            # split test row into input and output columns
            testX = test[i, :-1]
            # fit model on history and make a prediction
            # transform list into array
            train = np.asarray(train)
            # split into input and output columns
            trainX, trainy = train[:, :-1], train[:, -1]
            # fit model
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000)
            model.fit(trainX, trainy)
            # make a one-step prediction
            pred = model.predict([testX])[0]
            # store the prediction in the list of forecasts
            forecast.append(pred)

        return forecast


        
    def plot_predictions(self, predictions):
        plt.plot(self.train)
        plt.plot(self.test)
        plt.plot(predictions)
        plt.title('Predictions vs Actual Values')
        plt.legend(['Train', 'Test', 'Forecast'], loc='upper left')
        plt.ylabel('values')
        plt.xlabel('time')

if __name__ == "__main__":
 
    import pandas as pd
    #Create a dataframe of with two time series columns
    df = pd.DataFrame({"Date": ['01-01-2000', '01-02-2000', '01-03-2000', '01-04-2000','01-05-2000','01-06-2000','01-07-2000','01-08-2000','01-09-2000'] , "Close": [1,2,9,4,5,6,7,15,9], "Open": [2,3,4,14,6,7,8,9,10], "High": [3,4,5,8,7,8,11,10,7], "Low": [4,5,6,7,8,9,10,14,13]})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    df = df.append(df.copy())
    df = df.append(df.copy())
    df = df.append(df.copy())

    print(df)
    print(df.columns.tolist())
    #Initialise the class
    model = xgboost(df = df[["Close","Open","High","Low"]], target = "Close", test_size = 0.1)
    #Forecast the time series data
    forecast, test, train = model.fit()
    print("Forecast: ", forecast, "\n Test: ", test, "\n Train: ", train)
    print(forecast.shape)
    print(test.shape)



