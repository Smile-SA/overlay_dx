import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import cupy as cp
import cudf
from cuml.linear_model import LinearRegression as cuLinearRegression
from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
from cuml.neighbors import KNeighborsRegressor as cuKNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet, 
    HuberRegressor, SGDRegressor
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor
)
from joblib import parallel_backend, Parallel, delayed

class GPURegressorGridSearch:
    def __init__(self, use_gpu=True, n_jobs=-1):
        """
        Initialize GPU-accelerated regressor grid search
        
        Parameters:
        use_gpu (bool): Whether to use GPU acceleration when available
        n_jobs (int): Number of parallel jobs for grid search
        """
        self.use_gpu = use_gpu
        self.n_jobs = n_jobs
        self.models = {}
        self.grid_searches = {}
        self.best_params = {}
        self.scores = {}
        
        # Define models and their parameter grids
    
        self._initialize_models()

    def _initialize_models(self):
        """Initialize all regressors and their parameter grids"""
        
        # GPU-accelerated models (using RAPIDS cuML)
        if self.use_gpu:
            self.models['cuLinearRegression'] = {
                'model': cuLinearRegression(),
                'params': {
                    'fit_intercept': [True, False]
                },
                'is_gpu': True
            }
            
            self.models['cuRandomForest'] = {
                'model': cuRandomForestRegressor(),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, 30, None],
                    'max_features': ['auto', 'sqrt']
                },
                'is_gpu': True
            }
            
            self.models['cuKNN'] = {
                'model': cuKNeighborsRegressor(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                },
                'is_gpu': True
            }
            
        # CPU models with GPU support
        self.models['XGBoost'] = {
            'model': xgb.XGBRegressor(tree_method='gpu_hist' if self.use_gpu else 'hist'),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 0.9, 1.0]
            },
            'is_gpu': False
        }
        
        self.models['LightGBM'] = {
            'model': lgb.LGBMRegressor(device='gpu' if self.use_gpu else 'cpu'),
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.3],
                'num_leaves': [31, 62, 127]
            },
            'is_gpu': False
        }
        
        # CPU-only models
        cpu_models = {
            'Ridge': (Ridge(), {
                'alpha': [0.1, 1.0, 10.0],
                'solver': ['auto', 'svd', 'cholesky']
            }),
            'Lasso': (Lasso(), {
                'alpha': [0.1, 1.0, 10.0],
                'max_iter': [1000, 2000]
            }),
            'ElasticNet': (ElasticNet(), {
                'alpha': [0.1, 1.0, 10.0],
                'l1_ratio': [0.2, 0.5, 0.8]
            }),
            'GradientBoosting': (GradientBoostingRegressor(), {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6]
            })
        }
        
        for name, (model, params) in cpu_models.items():
            self.models[name] = {
                'model': model,
                'params': params,
                'is_gpu': False
            }

    
    def prepare_data(self, X, y, test_size=0.2):
        """
        Prepare data for GPU training
        """
        if self.use_gpu:
            if not isinstance(X, cudf.DataFrame):
                X = cudf.DataFrame(X)
            if not isinstance(y, cudf.Series):
                y = cudf.Series(y)
        return X, y
    
    def _to_gpu(self, X, y=None):
        """Convert data to GPU format"""
        if isinstance(X, (pd.DataFrame, np.ndarray)):
            X_gpu = cudf.DataFrame(X)
        else:
            X_gpu = X
            
        if y is not None:
            if isinstance(y, (pd.Series, np.ndarray)):
                y_gpu = cudf.Series(y)
            else:
                y_gpu = y
            return X_gpu, y_gpu
        return X_gpu

    def _to_cpu(self, X, y=None):
        """Convert data to CPU format"""
        if isinstance(X, cudf.DataFrame):
            X_cpu = X.to_pandas()
        else:
            X_cpu = X
            
        if y is not None:
            if isinstance(y, cudf.Series):
                y_cpu = y.to_pandas()
            else:
                y_cpu = y
            return X_cpu, y_cpu
        return X_cpu

    def fit_grid_search(self, X_train, y_train, X_test, y_test, scoring='neg_mean_squared_error'):
        """Perform grid search for all models"""
        self.results = {}
        
        for name, model_info in self.models.items():
            print(f"\nPerforming grid search for {name}...")
            
            # Convert data based on model type
            if model_info['is_gpu']:
                X_train_model, y_train_model = self._to_gpu(X_train, y_train)
                X_test_model, y_test_model = self._to_gpu(X_test, y_test)
            else:
                X_train_model, y_train_model = self._to_cpu(X_train, y_train)
                X_test_model, y_test_model = self._to_cpu(X_test, y_test)
            
            # Configure grid search
            grid_search = GridSearchCV(
                estimator=model_info['model'],
                param_grid=model_info['params'],
                scoring=scoring,
                n_jobs=self.n_jobs,
                cv=5,
                verbose=1
            )
            
            # Fit grid search
            with parallel_backend('threading', n_jobs=self.n_jobs):
                grid_search.fit(X_train_model, y_train_model)
            
            # Store results
            self.grid_searches[name] = grid_search
            self.best_params[name] = grid_search.best_params_
            
            # Make predictions
            y_pred = grid_search.predict(X_test_model)
            
            # Convert predictions and test data to CPU for metric calculation
            if model_info['is_gpu']:
                y_pred = y_pred.to_numpy()
                y_test_model = y_test_model.to_numpy()
            
            # Calculate metrics
            self.results[name] = {
                'mae': mean_absolute_error(y_test_model, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test_model, y_pred)),
                'r2': r2_score(y_test_model, y_pred),
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_
            } 
    

    def plot_results(self):
        """
        Plot performance comparison of all models
        """
        metrics = ['mae', 'rmse', 'r2']
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 5*len(metrics)))
        
        for i, metric in enumerate(metrics):
            scores = [self.results[model][metric] for model in self.results.keys()]
            axes[i].bar(self.results.keys(), scores)
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_xlabel('Models')
            axes[i].set_ylabel(metric.upper())
            plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
        
    def get_best_model(self):
        """
        Return the best performing model based on RMSE
        """
        best_model = min(self.results.items(), 
                        key=lambda x: x[1]['rmse'])
        return best_model[0], best_model[1]
    
    def save_results(self, filename='regressor_results.csv'):
        """
        Save results to CSV file
        """
        results_df = pd.DataFrame(self.results).T
        results_df.to_csv(filename)
        return results_df