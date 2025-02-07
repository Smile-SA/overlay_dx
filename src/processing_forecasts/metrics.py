"""Time series forecast evaluation metrics and methods"""
# Import libraries
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math

class Evaluate:
    """Evaluate: this class contains different mertics that can be used for time series forecast evaluation \n
    
    Methods: \n
        mae: mean absolute error \n
        mape: mean absolute percentage error \n
        rmse: root mean squared error \n
        mase: mean absolute scaled error \n
        evaluate: evaluate the model \n
        spikes_covering: compute the proportion of local extremes covered by the forecast \n
        overlay_dx_visualisation_df: computes the overlay_dx metric \n

    Import and usage: \n
        from src.processing_forecasts.metrics import Evaluate \n
        metrics = Evaluate(target_values = df["target_values"], prediction = df["prediction"]) \n
        metrics.evaluate() \n
        metrics.spikes_covering() \n
        results, aeras = metrics.overlay_dx_visualisation_df(df,100,0.1,0.1) \n

    """

    def __init__(self, target_values, prediction):
        """init: initialize the class"""
        self.target_values = target_values
        self.prediction = prediction

    def mae(self):
        """mae: mean absolute error"""
        return mean_absolute_error(self.target_values, self.prediction)
    
    def mape(self):
        """mape: mean absolute percentage error"""
        return mean_absolute_percentage_error(self.target_values, self.prediction)
    
    def rmse(self):
        """rmse: root mean squared error"""
        return np.sqrt(mean_squared_error(self.target_values, self.prediction))
    
    def mase(self):
        """mase: mean absolute scaled error"""
        # compute the naive forecast
        naive_forecast = self.target_values[:-1]
        # compute the naive error
        naive_error = mean_absolute_error(self.target_values[1:], naive_forecast)
        # compute the mase
        mase = self.mae()/naive_error
        return mase
    
    def evaluate(self):
        """evaluate: evaluate the model"""
        print("MAE: ", self.mae())
        print("MAPE: ", self.mape())
        print("RMSE: ", self.rmse())
        print("MASE: ", self.mase())

    def spikes_covering(self):
        """spikes_covering: compute the proportion of local extremes covered by the forecast"""
        # differentiating the target values
        diff = np.diff(self.target_values)
        print(diff)
        # Find indices of all maxima and minima
        maxima_idx = np.where((np.hstack([diff, 0]) < 0) & (np.hstack([0, diff]) > 0))[0]
        minima_idx = np.where((np.hstack([diff, 0]) > 0) & (np.hstack([0, diff]) < 0))[0]
        # Calculate the difference between the forecast and actual values at local maximums
        diff_max = self.prediction[maxima_idx] - self.target_values[maxima_idx]
        # Calculate the difference between the forecast and actual values at local minimums
        diff_min = self.prediction[minima_idx] - self.target_values[minima_idx]    
        # Compute the proportion of local extremes covered by the forecast
        max_covered = (diff_max >= 0).sum() / len(maxima_idx)
        min_covered = (diff_min <= 0).sum() / len(minima_idx)
        prop_covered = (max_covered + min_covered) / 2
        return prop_covered

    def overlay_dx(self,x,y=None):
        #TODO : what if x = 0 ?????
        ## ANSWER : just give a default constant value for now , lets go for x = 1 
        if x == 0:
            x = 1
        """overlay_dx computes the percentage of values where the absolute difference between the forecast and actual values is less than or equal to x"""
        if y is None:
            y = self.prediction
        # Calculate the absolute difference between the forecast and actual values
        abs_diff = abs(self.target_values-y)
        # Count the number of values where the absolute difference is less than or equal to 0.1
        num_overlay = (abs_diff <= x).sum()
        # Calculate the percentage of values that overlay
        pct_overlay = 100 * num_overlay / len(y)
        return pct_overlay

    def calculate_overlay_percentages_multi(
        self, forecasts, max_percentage, min_percentage, step,
        ):
        """calculate_overlay_percentages_multi: calculate the overlay percentages for multiple forecasts
        
        Args:
            forecasts (dict): dictionary of forecasts, keys are the names of the forecasts (str), values are the forecasts (np.array)
            max_percentage (int): maximum percentage of the range to calculate
            min_percentage (int): minimum percentage of the range to calculate
            step (int): step size for the percentages
        """

        # Calculate the range of the target values
        value_range = np.max(self.target_values) - np.min(self.target_values)
        
        # Define the percentages to calculate
        percentages = np.arange(max_percentage, min_percentage, -step)
        
        # Initialize an empty dictionary to store the results
        results = {}
        
        # Initialize an empty array to store the curve values
        curve_values = []
        
        # Calculate the overlay percentage for each percentage value for each forecast
        for name, forecast in forecasts.items():
            overlay_pct_values = []
            for pct in percentages:
                # Calculate the x value corresponding to the percentage of the range
                x = pct / 100 * value_range / 2
                
                # Calculate the overlay percentage using the overlay_x function
                overlay_pct = self.overlay_dx(x, self.target_values, forecast)
                overlay_pct_values.append(overlay_pct)
            
            # Store the overlay percentages for this forecast in the results dictionary
            results[name] = overlay_pct_values
            
            # Add the overlay percentages to the curve values array
            curve_values.append(overlay_pct_values)
        
        # Plot the overlay percentages for each forecast
        for name, overlay_pct_values in results.items():
            plt.plot(list(reversed(percentages)), list(reversed(overlay_pct_values)), label=name)      
        plt.xlabel("Percentage of range")
        plt.ylabel("Overlay percentage")
        plt.title("Overlay percentage vs. percentage of range")
        plt.ylim(0, 110)
        plt.xlim(max_percentage, min_percentage)
        plt.legend()
        
        plt.show()
        
        # Compute the area under the curve for each forecast
        areas = {}
        for name, curve in zip(forecasts.keys(), curve_values):
            area = np.trapz(curve, dx=step/100*value_range/2)
            max_area = value_range * (max_percentage - min_percentage) / 200
            percentage_covered = area / max_area
            areas[name] = percentage_covered
            
        # Return the results dictionary and the areas dictionary
        return results, areas
    
    def overlay_dx_visualisation_df(self, forecasts_df, max_percentage, min_percentage, step,
        save_in_file=False,
        saving_path = None):
        """calculate_overlay_percentages_multi: calculate the overlay percentages for multiple forecasts

        Args:
            forecasts_df (pd.DataFrame): DataFrame of forecasts, where columns are the names of the forecasts and rows are the forecast values
            max_percentage (int): maximum percentage of the range to calculate
            min_percentage (int): minimum percentage of the range to calculate
            step (int): step size for the percentages

        Description: 
            Overlay_dx is a score associated with a visualisation that provides a better understanding of the performance of our forecasts. Overlay_dx consists of several measures of the overlay metric, which draws an interval around the target values and returns the percentage of forecast values that fall within this interval. Overlay_dx calculates different measures of the overlay metric by reducing the size of its interval.  
                
            A score of 77% represents how well the forecasted values align with the actual values at different thresholds. It indicates that the achieved score is 77% of the maximum possible score, where perfect alignment would occur at all thresholds.  
            
            The score reflects the overall accuracy relative to the ideal scenario. The higher the score, the better the alignment between forecasted and actual values, while a lower score suggests larger deviations.  
            
            The overlay curve visualization is the key advantage of this metric. It allows for a quick assessment of model performance. By observing the curve, you can easily identify where significant deviations from the ideal scenario occur, helping to point areas for improvement.  
            
            Unlike other metrics, such as Mean Absolute Error, the overlay curve is less impacted by outliers, providing a more comprehensive view of accuracy.
            In summary, the overlay curve offers an intuitive and concise way to evaluate model performance, providing insights into accuracy across different thresholds and highlighting areas for optimization.
        """
            
        # Convert the DataFrame to a dictionary
        forecasts = forecasts_df.to_dict()
        # Calculate the range of the target values
        value_range = np.max(self.target_values) - np.min(self.target_values)

        # Define the percentages to calculate
        percentages = np.arange(max_percentage, min_percentage, -step)

        # Initialize an empty dictionary to store the results
        results = {}

        # Initialize an empty array to store the curve values
        curve_values = []

        # Calculate the overlay percentage for each percentage value for each forecast
        #for name, forecast in forecasts.items():
        #    print(len(forecasts_df[name]))
        #return
        for name, forecast in forecasts.items():
            forecast = forecasts_df[name]
            overlay_pct_values = []
            for pct in percentages:
                # Calculate the x value corresponding to the percentage of the range
                x = pct / 100 * value_range / 2  # TODO : REFACTOR ALL OF THIS , disambiguate this , check origin of value

                # Calculate the overlay percentage using the overlay_x function
                #print(len(forecast.values()))
                overlay_pct = self.overlay_dx(x,np.array(forecast))
                overlay_pct_values.append(overlay_pct)

            # Store the overlay percentages for this forecast in the results dictionary
            results[name] = overlay_pct_values

            # Add the overlay percentages to the curve values array
            curve_values.append(overlay_pct_values)

        # Plot the overlay percentages for each forecast
        for name, overlay_pct_values in results.items():
            plt.plot(list(reversed(percentages)), list(reversed(overlay_pct_values)), label=name)
        plt.xlabel("Percentage of range")
        plt.ylabel("Overlay percentage")
        plt.title("Overlay percentage vs. percentage of range")
        plt.ylim(0, 110)
        plt.xlim(max_percentage, min_percentage)
        plt.legend()
        if save_in_file:
            plt.savefig(saving_path)
        plt.show()
        
        # Compute the area under the curve for each forecast
        areas = {}
        for name, curve in zip(forecasts.keys(), curve_values):
            area = np.trapz(curve, dx=step / 100 * value_range / 2)
            max_area = value_range * (max_percentage - min_percentage) / 200
            percentage_covered = area / max_area
            areas[name] = percentage_covered

        # Return the results dictionary and the areas dictionary
        return results, areas

    def round_up(self,number, digits):
        factor = 10 ** digits
        return math.ceil(number * factor) / factor

    def overlay_dx_area_under_curve_metric(self,forecast,max_percentage,min_percentage,step):
         # Calculate the range of the target values
        value_range = np.max(self.target_values) - np.min(self.target_values)
        #TODO : what if value_range == 0 (constant func)

        # Define the percentages to calculate
        percentages = np.arange(max_percentage, min_percentage, -step)
        print(len(percentages))

        # Calculate the overlay percentage for each percentage value for each forecast
        overlay_pct_values = []
        for i, pct in enumerate(percentages):
            # Calculate the x value corresponding to the percentage of the range
            x = pct / 100 * value_range / 2  # TODO : REFACTOR ALL OF THIS , disambiguate this , check origin of value
            # Calculate the overlay percentage using the overlay_x function
            overlay_pct = self.overlay_dx(x,np.array(list(forecast)))
            overlay_pct_values.append(float(overlay_pct))
            #print(f"""
            #overlay input x : {x},
            #overlay percent values list : {overlay_pct_values}
            #overlay_value_this_iteration {i}: {overlay_pct}
#
            #""")

        # Store the overlay percentages for this forecast in the results dictionary

        # Add the overlay percentages to the curve values array
        curve = overlay_pct_values
        #print(f"""
        #curve : {curve},
        #percentages : {percentages} 
        #""")
        #area = np.trapz(curve, dx=step / 100 * value_range / 2)
        #max_area = value_range * (max_percentage - min_percentage) / 200
        dx = (percentages[0]-percentages[len(percentages)-1])/len(percentages)
        area = np.trapz(
            y=curve,
            dx=dx # to get from 0 to 100 eve though in reality it is reversed
        )
        max_area = max_percentage*100  #  xmax = max percentage  , ymax = 100 (max possible overlay value)
        #print(f"""
        #area : {area}
        #max area : {max_area}
        #""")
        percentage_covered = float(area / max_area)

        return float(self.round_up(percentage_covered,6))
 
    





    def overlay_dx_moo(self, forecasts_df, max_percentage, min_percentage, step):
        """
        overlay method for the multi objective approach
        """

        # Convert the DataFrame to a dictionary
        forecasts = forecasts_df.to_dict()

        # Calculate the range of the target values
        value_range = np.max(self.target_values) - np.min(self.target_values)

        # Define the percentages to calculate
        percentages = np.arange(max_percentage, min_percentage, -step)

        # Initialize an empty dictionary to store the results
        results = {}

        # Initialize an empty array to store the curve values
        curve_values = []

        # Calculate the overlay percentage for each percentage value for each forecast
        for name, forecast in forecasts.items():
            overlay_pct_values = []
            for pct in percentages:
                # Calculate the x value corresponding to the percentage of the range
                x = pct / 100 * value_range / 2

                # Calculate the overlay percentage using the overlay_x function
                overlay_pct = self.overlay_dx(x,np.array(list(forecast.values())))
                overlay_pct_values.append(overlay_pct)

            # Store the overlay percentages for this forecast in the results dictionary
            results[name] = overlay_pct_values

            # Add the overlay percentages to the curve values array
            curve_values.append(overlay_pct_values)

        # Compute the area under the curve for each forecast
        areas = {}
        for name, curve in zip(forecasts.keys(), curve_values):
            area = np.trapz(curve, dx=step / 100 * value_range / 2)
            max_area = value_range * (max_percentage - min_percentage) / 200
            percentage_covered = area / max_area
            areas[name] = percentage_covered

        # Return the results dictionary and the areas dictionary
        return results, areas

import unittest
class TestEvaluate(unittest.TestCase):

    def test_evaluate(self):
        # create the test data
        pred = np.array([2,1,2,3,2,1,2])
        forecast = np.array([2,1,2,3,2,1,2])
        test = Evaluate(pred, forecast)
        # tests for the evaluate class
        self.assertEqual(test.mae(), 0)
        self.assertEqual(test.mape(), 0)
        self.assertEqual(test.rmse(), 0)
        self.assertEqual(test.mase(), 0)
        self.assertEqual(test.spikes_covering(), 1)
        self.assertEqual(test.overlay_x(0.1), 100)

        # create new test data
        forecast = np.array([1,2,1,2,3,2,1])
        test = Evaluate(pred, forecast)

        # tests for the evaluate class
        self.assertEqual(test.mae(), 1)
        self.assertEqual(np.round(test.mape(),2), 0.62)
        self.assertEqual(test.rmse(), 1)
        self.assertEqual(test.mase(), 1.0)
        self.assertEqual(test.spikes_covering(), 0.0)
        self.assertEqual(test.overlay_x(0.1), 0.0)

if __name__ == '__main__':
    #unittest.main()
    import pandas as pd
    #Create a dataframe of with two time series columns
    df = pd.DataFrame({"Date": ['01-01-2000', '01-02-2000', '01-03-2000', '01-04-2000','01-05-2000','01-06-2000','01-07-2000','01-08-2000','01-09-2000'] , "true": [1,2,9,4,5,6,7,15,9], "lstm": [2,3,4,14,6,7,8,9,10], "var": [3,4,5,8,7,8,11,10,7], "cnn": [4,5,6,7,8,9,10,14,13]})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    print(df)
    metrics = Evaluate(df["true"], df["lstm"])
    results, aeras = metrics.overlay_dx_visualisation_df(df[["lstm"]], 100, 0, 0.1)
    print(aeras["lstm"])

    metrics.evaluate()
    


