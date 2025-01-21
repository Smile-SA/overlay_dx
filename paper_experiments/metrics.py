# module to put metrics here 
import os
import sys

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
sys.path.append(os.path.abspath('..'))
from src.processing_forecasts.metrics import Evaluate
from utils import measure_time


execution_time = {}
@measure_time(execution_times=execution_time)
def rmse_metric(y_true,y_pred):
    return root_mean_squared_error(y_true, y_pred)

@measure_time(execution_times=execution_time)
def mse_metric(y_true,y_pred):
    return mean_squared_error(y_true,y_pred)

@measure_time(execution_times=execution_time)
def mae_metric(y_true,y_pred):
    return mean_absolute_error(y_true,y_pred)

@measure_time(execution_times=execution_time)
def mape_metric(y_true,y_pred):
    return mean_absolute_percentage_error(y_true,y_pred)

   

@measure_time(execution_times=execution_time)
def wmape_metric(y_true,y_pred):
    ## needs to be implemented myself
    abs_diff = np.abs(y_true-y_pred)
    abs_actual = np.abs(y_true)
    if np.sum(abs_actual) == 0:
        return np.sum(abs_diff)/0.000000001 # our epsilon here

    # will do the exact formula on the paper ... tis is NOT std wmap
    return (np.sum(abs_diff)/np.sum(abs_actual))


@measure_time(execution_times=execution_time)
def nrmse_metric(y_true,y_pred):
    m = np.mean(y_true)
    maxx = np.max(y_true)
    minn = np.min(y_true)
    rang = maxx-minn
    if m!= 0:
        return root_mean_squared_error(y_true,y_pred)/m
    elif rang != 0:
        return root_mean_squared_error(y_true,y_pred)/rang

    else:
        # return normal rmse ....
        return root_mean_squared_error(y_true,y_pred)

@measure_time(execution_times=execution_time)
def aic_metric(y_true,y_pred):
    ## need to ccheck model params issue 
    # link : https://www.statsmodels.org/dev/generated/statsmodels.tools.eval_measures.aic.html
    pass

@measure_time(execution_times=execution_time)
def overlay_metric(y_true,y_pred):
    metrics_eval = Evaluate(y_true,y_pred)
    overlay_score = metrics_eval.overlay_dx_area_under_curve_metric(
                    y_pred,
                    100,
                    0,
                    0.1,
                )
    return overlay_score