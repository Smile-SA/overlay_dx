
# Configuration for Linear Regression
random_state = 30
LR_CONFIG = {
    'fit_intercept': True,
    'copy_X': True
}

# Configuration for Random Forest Regressor
RF_CONFIG = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'bootstrap': True,
    "random_state" : random_state
}

# Configuration for XGBoost Regressor
XGB_CONFIG = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 6,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    "random_state" : random_state
}

# Configuration for LightGBM Regressor
LGBM_CONFIG = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'num_leaves': 50,
    'max_depth': 10,
    'subsample_for_bin': 300000,
    "random_state" : random_state
}

# Configuration for K Neighbors Regressor
KNN_CONFIG = {
    'n_neighbors': 5,
    'weights': 'uniform',
    'algorithm': 'auto',
    'leaf_size': 30,
    #"random_state" : random_state
}

# Configuration for ARIMA
ARIMA_CONFIG = {
   "order" : (1,1,1)
}

# Configuration for FFT (Fourier Transform) in Time Series Analysis
FFT_CONFIG = {
    'n_components': 10,
    'threshold': 0.1
}

# Configuration for BaggingRegressor (replacing SVR)
BAGGING_CONFIG = {
    'n_estimators': 10,
    'max_samples': 1.0,
    'max_features': 1.0,
    'bootstrap': True,
    'bootstrap_features': False,
    'random_state': random_state
}

EXTRATREES_CONFIG = {
   'n_estimators': 100,
   'max_depth': None,
   'min_samples_split': 2,
   'min_samples_leaf': 1,
   'bootstrap': False,
   'n_jobs': -1,
   'random_state': random_state
}




regressors_configs = {
    "LR_CONFIG" : LR_CONFIG,
    "RF_CONFIG" : RF_CONFIG,
    "XGB_CONFIG" : XGB_CONFIG,
    "LGBM_CONFIG" : LGBM_CONFIG,
    "KNN_CONFIG" : KNN_CONFIG,
    "ARIMA_CONFIG" : ARIMA_CONFIG,
    "FFT_CONFIG" : FFT_CONFIG,
    "BAGGING_CONFIG" : BAGGING_CONFIG,
    "EXTRATREES_CONFIG" : EXTRATREES_CONFIG
}

