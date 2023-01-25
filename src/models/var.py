"""Vector Auto Regression, VAR is a multivariate forecasting algorithm that is used when two or more time series influence each other.
"""
#Importing libraries
import numpy as np
from statsmodels.tsa.api import VAR
import math
import numpy as np
import matplotlib.pyplot as plt

class var():
    '''This class is used to forecast and model time series data using VAR models. \n
    
    Parameters: \n
        df (pandas dataframe): pandas dataframe with the time series as columns \n
        target (str): name of target column to forecast \n
        test_size (float): the percentage size of the test set from 0 to 1 \n
        max_lag (int): the maximum number of lags to use in the VAR model \n
    
    Returns: \n
        dataframe: a dataframe with the original data and the modelled data in a new column with the name of the target + "_simulated" \n
    
    remark: the VAR model is not a forecasting model, it is a modelling tool. The VAR model is used to model the time series and then the model is used to forecast the time series. \n
    The VAR model is a multivariate model, it uses all the columns of the dataframe to model the target column. \n
    In fact with this implementation the var modelling is applied on all columns of the dataframe from the others. \n

    Import and usage: \n
        from src.models.var import var \n
        var_model = var(df, target, test_size, max_lag) \n
        forecast, test, train = var_model.fit() \n
    '''

    def __init__(self, df, target, test_size, max_lag):
        self.data = df
        self.target = target
        self.max_lag = max_lag
        self.test_size = math.ceil(test_size*self.data.shape[0])
        self.train = self.data[:-self.test_size]
        self.test = self.data[-self.test_size:]
        self.results_params = None
   
    def fit(self):
        """var main function"""
        results = self.run_var(self.train,self.max_lag)
        results_params = self.extract_var_results(results)
        result_df = self.extract_var_equations(self.data,results_params)

        # plot the results
        self.plot_predictions(result_df[self.target+"_simulated"][-self.test_size:])

        # return the forecast, test and train data
        return result_df[self.target+"_simulated"][-self.test_size:], self.test[self.target], self.train[self.target]
    

    def get_lag_series(self,col,results_params):
        """This function returns an array from the VAR results with 3 elements:
        name: name of the serie
        lag: delay
        col: the serie that we model
        """
        return results_params.loc[(results_params[col].isna() == False) & (results_params["lag"] != "o") ,("name",'lag',col)].values

    def run_var(self,data,max_lag):
        """Initialise and run VAR"""
        model = VAR(data)
        model.fit(max_lag)
        model.select_order(max_lag)
        results = model.fit(maxlags=max_lag, ic='aic')
        return results

    def extract_var_results(self,results):  
        """VAR results analysis and selection (statistical test)"""

        results_params = results.params
        results_pvalues = results.pvalues

        # create a mask of pvalues below 0.05
        mask = results_pvalues < 0.05
        mask.iloc[0,:] = True
        # use the mask to set params to 0 where pvalues are above 0.05
        results_params[mask == False] = np.nan
        results_params["lag"] = results_params.index.str[1]
        results_params["name"] = results_params.index.str[3:]
        self.results_params = results_params
        return results_params

    def extract_var_equations(self,data,results_params):
        """Extract the equations"""
        result_df = data.copy()
        #Loop over the time series
        for each in data.columns:
            result_df[each+"_simulated"]=0
            #loop over time
            lagged_series = self.get_lag_series(each,results_params)
                #loop over lagged series
            for array in lagged_series:
                result_df[each+"_simulated"] += data[array[0]].shift(int(array[1])) * float(array[2])
            result_df[each+"_simulated"] += results_params[each][0]
        return result_df
    
    def forecast(self,new_data):
        """forecast function: use the VAR model equation to forecast the time series
        
        Args: 
        new_data (pandas dataframe): dataframe with the same columns as the training data
        /!\ don't forget to add the max_lag number of rows before the first row of the new data that you want to forecast since we use the lagged values to forecast.
        """
        for each in new_data.columns:
            new_data[each+"_simulated"]=0
            #loop over time
            lagged_series = self.get_lag_series(each,self.results_params)
                #loop over lagged series
            for array in lagged_series:
                new_data[each+"_simulated"] += new_data[array[0]].shift(int(array[1])) * float(array[2])
            new_data[each+"_simulated"] += self.results_params[each][0]

        return  new_data
    
    def plot_predictions(self, predictions):
        plt.plot(self.train[self.target])
        plt.plot(self.test[self.target])
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
    print(df)
    print(df.columns.tolist())
    #Initialise the class
    model = var(df = df[["Close","Open","High","Low"]], target = "Close", test_size = 0.2, max_lag= 1)
    #Forecast the time series data
    forecast, test, train=model.fit()
    print("forecast",forecast, "test", test, "train", train)
    new_data = pd.DataFrame({"Date": ['01-10-2000', '01-11-2000', '01-12-2000'] , "Close": [np.nan,np.nan,np.nan], "Open": [11,12,13], "High": [12,13,14], "Low": [13,14,15]}).set_index("Date")
    forecast = model.forecast(new_data = new_data)
    print(forecast)