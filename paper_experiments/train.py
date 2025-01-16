import os
import sys

import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn import linear_model
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

from metrics import (
    rmse_metric,
    mse_metric,
    mae_metric,
    mape_metric,
    wmape_metric,
    nrmse_metric,
    aic_metric,
    overlay_metric
)

sys.path.append(os.path.abspath(".."))
from src.processing_forecasts.metrics import Evaluate



# make a function that include all regression models
def score_with_metric(X_test,y_test,regressor,metric):
    pred = regressor.predict(X_test)
    score = metric(y_test,pred)
    return score

def maeScore(regressor):
    """
    :param regressor: regressor model function
    :return: mean absolute error for regressor
    """
    clf = regressor.fit(X_train, y_train)
    pred = clf.predict(X_test)
    mae_score = mean_absolute_error(y_test, pred)
    return mae_score
# function for root mean square error


def rmseScore(regressor):
    """
    :param regressor: regressor model function
    :return: root mean score error for regressor
    """

    clf = regressor.fit(X_train, y_train)
    pred = clf.predict(X_test)
    rmse_score = np.sqrt(mean_squared_error(y_test, pred))
    return rmse_score


    # mean absolute error, root mean square error for each regressors
    for regressor in Regressors:
        name = regressor.__class__.__name__
        mae_dict[name] = maeScore(regressor)
        mae_scores.append(mae_dict[name])
        rmse_dict[name] = rmseScore(regressor)
        rmse_scores.append(rmse_dict[name])
        overlay_dict[name] = overlay_score(regressor)
        overlay_scores.append(overlay_dict[name])

        print(
            "*************************************************************************"
        )
        print(f"{name} Mean Absolute Error = {mae_dict[name]}")
        print(f"{name} Root Mean Square Error = {rmse_dict[name]}")
        print(f"{name} overlay similarity = {overlay_dict[name]}")

    # Plotting the performance of regressors
def allRegressors(X_train, X_test, y_train, y_test, run_id):
    """
    This function use multiple machine learning regressors and show us the results of them
    :param X_train: train input
    :param X_test: test input
    :param y_train: train output
    :param y_test: test output
    :return: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE) for each regressors and
    comparison plot for regressors according to RMSE and MAE
    """
    Regressors = [
        LinearRegression(),
        RandomForestRegressor(),
        XGBRegressor(),
        LGBMRegressor(),
        KNeighborsRegressor(),
        ARIMA()
    ]
    reg_config_to_reg = {
    "LR_CONFIG" : LinearRegression(),
    "RF_CONFIG" : RandomForestRegressor(),
    "XGB_CONFIG" :XGBRegressor(),
    "LGBM_CONFIG" :LGBMRegressor(),
    "KNN_CONFIG" :KNeighborsRegressor(),
    "ARIMA_CONFIG" :ARIMA(),
    }
    # add ARIMA   , FFT
    from regressors_configs import regressors_configs
    # ADd hyperparams and max number of jobs
    for config_name in Regressors:
        reg = reg_config_to_reg[config_name]
        reg.set_params(n_jobs=-1)
        params = regressors_configs[config_name]
        reg.set_params(**params)

    metrics =  {
        rmse_metric : "rmse",
        mse_metric : "mse",
        mae_metric  : "mae",
        mape_metric : "mape",
        wmape_metric : "wmape",
        nrmse_metric : "nrmse",
        aic_metric : "aic",
        overlay_metric : "overlay_metric"
    }
    res_dict = { metrics[metric] : {} for metric in metrics}
    for reg_conf in reg_config_to_reg:
        reg = reg_config_to_reg[reg_conf]
        name = reg.__class__.__name__
        reg.fit(X_train,y_train)
        for metric in metrics:
            score = score_with_metric(X_test=X_test,y_test=y_test,regressor=reg,metric=metric)
            res_dict[metrics[metric]] = # TODO: monctine tomorrow here .... 
        mae_dict[name] = maeScore(regressor)
        mae_scores.append(mae_dict[name])
        rmse_dict[name] = rmseScore(regressor)
        rmse_scores.append(rmse_dict[name])
        overlay_dict[name] = overlay_score(regressor)
        overlay_scores.append(overlay_dict[name])

        print(
            "*************************************************************************"
        )
        print(f"{name} Mean Absolute Error = {mae_dict[name]}")
        print(f"{name} Root Mean Square Error = {rmse_dict[name]}")
        print(f"{name} overlay similarity = {overlay_dict[name]}")



        
    
    



def plotPerformance(scores_list, scores_dict, metric: str):
    """
    :param scores_list: list that include evaluation scores
    :param scores_dict: dictionary that include regressors and evaluation scores
    :param metric: metric name y axis
    :return: plot of performance comparison of regressors
    """

    N = len(Regressors)
    w = 0.5
    x = np.arange(N)
    plt.bar(x, scores_list, width=w, align="center", color="g")
    plt.xlabel("Regressors")
    plt.title("Performance Comparison of Regressors")
    plt.ylabel(f"{metric} Error")
    plt.xticks(x, scores_dict.keys(), rotation=90)
    plt.yticks(
        np.arange(0, np.max(scores_list), np.max(scores_list) / len(scores_list))
    )
    plt.show()


def save_results(rmse, mae, overlay):
    df = pd.DataFrame(
        {
            "Model": rmse.keys(),
            "RMSE": rmse.values(),
            "MAE": mae.values(),
            "Overlay": overlay.values(),
        }
    )
    df.to_csv(f"metrics_run{run_id}.csv")
    return df


# call the functions
save_results(rmse_dict, mae_dict, overlay_dict)
plotPerformance(mae_scores, mae_dict, "Mean Absolute")
plotPerformance(rmse_scores, rmse_dict, "Root Mean Square")
plotPerformance(overlay_scores, overlay_dict, "overlay similarity")
