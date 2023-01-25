"""Ensemble learning methods can be used for improoving forecasting models"""
import pandas as pd
import numpy as np
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.ensemble import StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


class ensemble_learning():
    """ This class embeds the ensemble learning methods: \n

    Methods: \n
    - Bagging: multiple independent models are trained on different subsets of the training data, and their predictions are combined through averaging, voting or weighted averaging (Random Forest). \n
    - Boosting: multiple models are trained sequentially, with each new model attempting to correct the errors made by the previous models (Gradient Boosting algorithm).\n
    - Stacking: multiple models with different characteristics are trained and their predictions are combined using a meta-model(Decision Tree Regressor).  \n

    Attributes: \n
    - forecasts (list) : list of the forecasts to use \n
    - validation_data (numpy array) : numpy array of the validation data \n

    Import and use: \n
        from src.processing_forecasts.ensemble_learning import ensemble_learning \n
        ensemble_learning = ensemble_learning(forecasts, validation_data) \n
        bagged_forecast = ensemble_learning.bagging() \n
        boosted_forecast = ensemble_learning.boosting() \n
        stacked_forecast = ensemble_learning.stacking() \n
    """

    def __init__(self, forecasts, validation_data):
        self.forecasts = forecasts
        self.EL_test_size = math.ceil(0.3*(validation_data).shape[0])
        self.combined_forecasts = np.column_stack(forecasts)
        self.train_data = self.combined_forecasts[:-self.EL_test_size]
        self.X_train, self.y_train = self.combined_forecasts[:-self.EL_test_size], validation_data[:-self.EL_test_size]
        self.X_test, self.y_test = self.combined_forecasts[-self.EL_test_size:], validation_data[-self.EL_test_size:]

    def bagging(self):
        """ This method implements the bagging ensemble method. \n
        """

        # Define hyperparameter grid for grid search
        param_grid = {
            'n_estimators': [10, 50, 100, 200, 300, 400],
            'max_depth': [None, 5, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10]
        }

        # Perform grid search to find the best random forest regressor
        rf = RandomForestRegressor()
        grid_search = GridSearchCV(rf, param_grid=param_grid, cv=5)
        
        # Combine training and testing data for scaling
        X_combined = np.vstack((self.X_train, self.X_test))
        y_combined = np.concatenate((self.y_train, self.y_test))
        
        # Scale the combined features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        
        # Perform cross-validation to obtain bagged forecast
        bagging_forecast = cross_val_predict(grid_search, X_scaled, y_combined, cv=5)
        
        # Split the bagged forecast into training and testing predictions
        bagging_train_pred = bagging_forecast[:len(self.X_train)]
        bagging_test_pred = bagging_forecast[len(self.X_train):]
        
        # Evaluate the performance of the bagging ensemble
        bagging_score = mean_squared_error(self.y_test, bagging_test_pred)

        return bagging_test_pred, self.y_test, bagging_score



    def boosting(self):
        """ This method implements the boosting ensemble method. \n
        """

        # Define hyperparameter grid for grid search
        param_grid = {
            'n_estimators': [10, 50, 100, 200, 300, 400],
            'learning_rate': [0.1, 0.01, 0.001],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }

        # Perform grid search to find the best gradient boosting regressor
        gb = GradientBoostingRegressor()
        grid_search = GridSearchCV(gb, param_grid=param_grid, cv=5)
        grid_search.fit(self.X_train, self.y_train)
        best_gb = GradientBoostingRegressor(**grid_search.best_params_)

        # Train the best gradient boosting model on the training data
        best_gb.fit(self.X_train, self.y_train)

        # Generate new forecasts using the trained model
        boosting_forecast = best_gb.predict(self.X_test)

        # Evaluate the performance of the boosting ensemble
        boosting_score = mean_squared_error(self.y_test, boosting_forecast)

        return boosting_forecast, self.y_test, boosting_score

    
    def stacking(self):
        """ This method implements the stacking ensemble method. \n
        """
       # Combine individual regressors and the final meta-regressor
        regressors = [
            ('regressor1', RandomForestRegressor()),
            ('regressor2', SVR()),
            ('regressor3', GradientBoostingRegressor())
        ]
        stack_reg = StackingRegressor(estimators=regressors, final_estimator=DecisionTreeRegressor())

        # Combine the training and testing sets
        X_combined = np.vstack((self.X_train, self.X_test))
        y_combined = np.concatenate((self.y_train, self.y_test))

        # Scale the combined features
        X_scaled = StandardScaler().fit_transform(X_combined)

        # Perform cross-validation
        stack_forecasts = cross_val_predict(stack_reg, X_scaled, y_combined, cv=5)

        # Split the forecasts into training and testing predictions
        stack_train_pred = stack_forecasts[len(self.X_train):]
        stack_test_pred = stack_forecasts[:len(self.X_test)]

        # Evaluate the performance of the stacking ensemble
        stacking_score = mean_squared_error(self.y_test, stack_test_pred)

        return stack_test_pred, self.y_test, stacking_score
    

if __name__ == '__main__':
    list_of_arrays = [np.array([1, 2, 3,1, 2, 3,1, 2, 3,1, 2, 3]), np.array([4, 5, 6,4, 5, 6,4, 5, 6,4, 5, 6]), np.array([7, 8, 9,7, 8, 9,7, 8, 9,7, 8, 9])]
    model = ensemble_learning("boosting", list_of_arrays, np.array([3,4,5,3,4,5,3,4,5,3,4,5]))
    print(model.boosting())
