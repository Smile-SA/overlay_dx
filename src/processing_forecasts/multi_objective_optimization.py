"""
Multi objective Optimization can find the best weighted average in regard of multiple metrics to minimize and scores to maximize.
"""
import numpy as np
import pandas as pd
from platypus.problems import Problem
from platypus.algorithms import NSGAII
from platypus.types import Real
from platypus.core import nondominated
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from .metrics import Evaluate
#local testing
#from metrics import Evaluate

class MyProblem(Problem):
    """MyProblem: class to define the multi-objective optimization problem \n
        Three objectives: \n
            1. Minimize the MAE \n
            2. Minimize the RMSE \n
            3. Maximize the overlay_dx_score \n
        Find the best weights for the weighted average forecast according to the three objectives. \n

    how it works: https://platypus.readthedocs.io/en/latest/getting-started.html#defining-a-problem \n

    Attributes: \n
        df (pd.DataFrame): pandas dataframe with the forecasts and the target values \n
        target_values (np.array): target values \n
        forecasts (dict): dictionary of forecasts, keys are the approach names of the forecasts (str), values are the forecasts (np.array) \n
    
    Import and usage: \n
        from src.processing_forecasts.multi_objective_optimization import find_best_weights \n
        best_weights , best_result = find_best_weights(df, forecasts_cols, target_col, max_evaluations) \n

    """

    def __init__(self, df, forecasts_cols,target_col,max_evaluations):
        """Initialise the class with the attributes"""
        super().__init__(len(forecasts_cols), len(forecasts_cols))
        self.df = df
        self.types[:] = Real(0, 1)
        self.directions[0] = Problem.MINIMIZE  # minimize the loss
        self.directions[1] = Problem.MINIMIZE  # minimize the loss
        self.directions[2] = Problem.MAXIMIZE  # maximize the gains
        self.constraints[:] = "==0"
        self.target_col = target_col
        self.forecasts_cols = forecasts_cols
        self.max_evaluations = max_evaluations

    def evaluate(self, solution):
        """Evaluate the solution"""
        variables = solution.variables
        loss_1 = self.rmse_loss_function(variables)
        loss_2 = self.mae_loss_function(variables)
        gain = self.overlay_dx_gain_function(variables)

        if isinstance(gain, list):
            gain = gain[0]  # Adjust this based on your actual needs

        solution.objectives[:] = [loss_1, loss_2, gain]
        solution.constraints[:] = [sum(variables) - 1]

    
    def rmse_loss_function(self,weights):
        """Compute the RMSE between the actual data and the weighted average forecast"""
        # Compute the weighted average of the forecasts
        forecast = np.zeros_like(self.df[self.target_col])
        for i, col in enumerate(self.df[self.forecasts_cols]):
            forecast += self.df[col] * weights[i]
        # Compute the MAE between the actual data and the weighted average forecast
        rmse = mean_squared_error(self.df[self.target_col], forecast)
        return rmse
    
    def mae_loss_function(self,weights):
        """Compute the MAE between the actual data and the weighted average forecast"""
        # Compute the weighted average of the forecasts
        forecast = np.zeros_like(self.df[self.target_col])
        for i, col in enumerate(self.df[self.forecasts_cols]):
            forecast += self.df[col] * weights[i]
        # Compute the MAE between the actual data and the weighted average forecast
        mae = mean_absolute_error(self.df[self.target_col], forecast)
        return mae

    def overlay_dx_gain_function(self, weights):
        """Compute the overlay_dx_score between the actual data and the weighted average forecast"""
        # Compute the weighted average of the forecasts
        forecast = np.zeros_like(self.df[self.target_col])
        for i, col in enumerate(self.df[self.forecasts_cols]):
            forecast += self.df[col] * weights[i]
        
        #print("forecast",forecast)
        #print("target",self.df)
        metrics = Evaluate( target_values = self.df[self.target_col], prediction= forecast)
        
        # convert forecast to pandas series with the name "forecast"
        forecast = pd.Series(forecast, name="forecast")
        forecast = forecast.to_frame()
        #print("forecast",forecast)

        pct_overlay = metrics.overlay_dx_moo(forecast,100,0.1,0.1)[1]
        #print("pct_overlay",pct_overlay["forecast"])

        return pct_overlay["forecast"]
    
def find_best_weights(df, forecasts_cols,target_col,max_evaluations):
    """find_best_weights: find the best weights for the weighted average of forecasts according to the three objectives.
        Three objectives:
            1. Minimize the MAE
            2. Minimize the RMSE
            3. Maximize the overlay_dx_score
        Find the best weights for the weighted average forecast according to the three objectives.

    how it works: https://platypus.readthedocs.io/en/latest/getting-started.html#defining-a-problem

    Args:
        df (pd.DataFrame): pandas dataframe with the forecasts and the target values
        num_variables (int): number of variables
        target_values (np.array): target values
        forecasts (dict): dictionary of forecasts, keys are the approach names of the forecasts (str), values are the forecasts (np.array)
        
    Import: from src.processing_forecasts.multi_objective_optimization import find_best_weights
      """
    problem = MyProblem(df, forecasts_cols,target_col,max_evaluations)
    optimizer = NSGAII(problem)
    optimizer.run(max_evaluations)
    pareto_front = nondominated(optimizer.result)
    best_solution = pareto_front[0]
    best_weights = best_solution.variables
    best_results = {'RMSE': best_solution.objectives[0], 'MAE': best_solution.objectives[1], 'Overlay_dx': best_solution.objectives[2]}  # negate the gain objectives back to positive values
    return best_weights, best_results
    

if __name__ == "__main__":

    import pandas as pd
    #Create a dataframe of with two time series columns
    df = pd.DataFrame({"Date": ['01-01-2000', '01-02-2000', '01-03-2000', '01-04-2000','01-05-2000','01-06-2000','01-07-2000','01-08-2000','01-09-2000'] , "true": [1,2,9,4,5,6,7,15,9], "lstm": [2,3,4,14,6,7,8,9,10], "var": [3,4,5,8,7,8,11,10,7], "cnn": [4,5,6,7,8,9,10,14,13]})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")
    print(df)
    #Initialise the class
    #Run the optimization
    best_weights, best_results = find_best_weights(df, ["lstm","var","cnn"] ,"true",10)

    print(best_weights)
    print(best_results)

