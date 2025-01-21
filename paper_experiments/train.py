import pandas as pd
import pickle
import os
import sys
import copy as cp
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
from statsmodels.tsa.arima.model import ARIMAResultsWrapper
#from statsmodels.tsa.arima.model import ARIMA
#from pmdarima import AutoARIMA

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
sys.path.append(os.path.abspath('..'))

from utils import measure_time

# make a function that include all regression models
def score_with_metric(X_test,y_test,regressor,metric):
    print(regressor)
    if isinstance(regressor,ARIMAResultsWrapper):
        pred = regressor.forecast(len(y_test))
        score = metric(y_test,pred)
    else:
        pred = regressor.predict(X_test)
        score = metric(y_test,pred)
    return score

    


def allRegressors(X_train, X_test, y_train, y_test, run_name,saving_path):
    """
    This function use multiple machine learning regressors and show us the results of them
    :param X_train: train input
    :param X_test: test input
    :param y_train: train output
    :param y_test: test output
    :return: Root Mean Squared Error (RMSE), Mean Absolute Error (MAE) for each regressors and
    comparison plot for regressors according to RMSE and MAE
    """
    #Regressors = [
    #    LinearRegression(),
    #    RandomForestRegressor(),
    #    XGBRegressor(),
    #    LGBMRegressor(),
    #    KNeighborsRegressor(),
    #]
    reg_config_to_reg = {
    "LR_CONFIG" : LinearRegression(),
    "RF_CONFIG" : RandomForestRegressor(),
    "XGB_CONFIG" :XGBRegressor(),
    "LGBM_CONFIG" :LGBMRegressor(),
    "KNN_CONFIG" :KNeighborsRegressor(),
    "ARIMA_CONFIG" : [],
    }
    # add ARIMA   , FFT
    from regressors_config import regressors_configs
    # ADd hyperparams and max number of jobs
    for config_name in reg_config_to_reg:
        if config_name == "ARIMA_CONFIG":
            continue
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
        #aic_metric : "aic",
        overlay_metric : "overlay_metric"
    }
    res_dict = { metrics[metric] : {} for metric in metrics}
    pred_dict = { metrics[metric] : {} for metric in metrics}
    for reg_conf in reg_config_to_reg:
        if reg_conf=="ARIMA_CONFIG":
            reg_config_to_reg[reg_conf] = ARIMA(
                endog=y_train,
                order = (2,1,1),
                trend="t"
            ).fit()
            reg = reg_config_to_reg[reg_conf]
            name = "ARIMA"
        else:
            reg = reg_config_to_reg[reg_conf]
            name = reg.__class__.__name__
            reg.fit(X_train,y_train)
        for metric in metrics:
            #print(reg)
            score = score_with_metric(X_test=X_test,y_test=y_test,regressor=reg,metric=metric)
            res_dict[metrics[metric]][name] = score
            if reg_conf == "ARIMA_CONFIG":
              pred_dict[metrics[metric]][name] = {
                "y_true" : cp.deepcopy(y_test),
                "y_pred" : list(reg.forecast(len(y_test)))
                }
            else:
                pred_dict[metrics[metric]][name] = {
                "y_true" : cp.deepcopy(y_test),
                "y_pred" : reg.predict(X_test)
                }
            print(f"y_true shape {y_test.shape} ")
            print(f"{name} with metric {metrics[metric]} = {score}")
        print("***************************************************************************")
    #print(res_dict)
    #print(os.path.join(saving_path,f"results_run_{run_name}.pkl"))
    #with open(os.path.join(saving_path,f"results_run_{run_name}.pkl"),"wb") as f:
    #    pickle.dump(res_dict,f)
    df_met = pd.DataFrame(res_dict)
    df_met.to_csv(os.path.join(saving_path,f"results_run{run_name}.csv"))
    from metrics import execution_time
    df_execution= pd.DataFrame(execution_time)
    df_execution.to_csv(os.path.join(saving_path,f"execution_time{run_name}.csv"))
    print(execution_time)
    # dump regressors
    with open(os.path.join(saving_path,f"regressors_run{run_name}") ,"wb") as f:
        pickle.dump(reg_config_to_reg,f)
    return df_met, df_execution , pred_dict

 

