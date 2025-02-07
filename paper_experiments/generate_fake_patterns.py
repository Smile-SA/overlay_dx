import copy as cp
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.abspath('..'))
from metrics import (
    aic_metric,
    mae_metric,
    mape_metric,
    mse_metric,
    nrmse_metric,
    overlay_metric,
    rmse_metric,
    wmape_metric,
)
from tqdm import tqdm

from src.processing_forecasts.metrics import Evaluate

np.random.seed(42) # set up random seed for reproducibility
def generate_baseline_series(n_points=1000, freq='D'):
    """
    Generate multiple baseline time series cases.
    
    Args:
        n_points: Number of data points
        freq: Frequency for date range ('D' for daily, 'H' for hourly, etc.)
    """
    date_rng = pd.date_range(start='2023-01-01', periods=n_points, freq=freq)
    x = np.linspace(0, 4*np.pi, n_points)
    
    # Generate different baseline patterns
    baselines = {
        'constant': np.ones(n_points) * 100,
        'linear_trend': np.linspace(0, 100, n_points),
        'exponential_trend': np.exp(np.linspace(0, 4.6, n_points)),
        'seasonal': 50 + 30 * np.sin(x),
        'seasonal_with_trend': 50 + 30 * np.sin(x) + np.linspace(0, 50, n_points),
        'multiple_seasonality': 50 + 20 * np.sin(x) + 10 * np.sin(2*x),
        'random_walk': np.cumsum(np.random.normal(0, 0.1, n_points))
    }
    
    # Convert to DataFrame
    df = pd.DataFrame(baselines, index=date_rng)
    
    # Add noise to each series
    for col in df.columns:
        noise = np.random.normal(0, df[col].std()*0.05, n_points)
        df[f'{col}_with_noise'] = df[col] + noise
    return df

def generate_baseline_and_predictions(n_points=1000, freq='D'):
    """Generate baselines and corresponding predictions with controlled deviations."""
    date_rng = pd.date_range(start='2023-01-01', periods=n_points, freq=freq)
    x = np.linspace(0, 4*np.pi, n_points)
   
    patterns = {
        'constant': (np.ones(n_points) * 100, {
            'perfect': lambda x: x,
            'noisy': lambda x: x + np.random.normal(0, 5, n_points),
            'biased': lambda x: x * 1.1,
            'delayed': lambda x: np.roll(x, 5),
            'outliers': lambda x: x + np.where(np.random.random(n_points) > 0.98, 50, 0)
        }),
        
        'linear_trend': (np.linspace(0, 100, n_points), {
            'perfect': lambda x: x,
            'underestimate': lambda x: x * 0.8,
            'overestimate': lambda x: x * 1.2,
            'lagged': lambda x: np.roll(x, 10),
            'nonlinear_bias': lambda x: x + 0.001 * x**2,
            'step_changes': lambda x: x + np.where(x > 50, 20, 0)
        }),
        
        'seasonal': (50 + 30 * np.sin(x), {
            'perfect': lambda x: x,
            'amplitude_error': lambda x: 50 + 20 * np.sin(x),
            'phase_shift': lambda x: 50 + 30 * np.sin(x + 0.5),
            'missed_peaks': lambda x: np.where(x > 60, 60, x),
            'frequency_error': lambda x: 50 + 30 * np.sin(1.1 * x),
            'asymmetric_error': lambda x: x + np.where(np.sin(x) > 0, 5, -2)
        }),
        
        'random_walk': (np.cumsum(np.random.normal(0, 0.1, n_points)), {
            'perfect': lambda x: x,
            'smoothed': lambda x: pd.Series(x).rolling(5).mean().fillna(method='bfill'),
            'delayed': lambda x: np.roll(x, 3),
            'noisy': lambda x: x + np.random.normal(0, x.std()*0.1, n_points),
            'trend_biased': lambda x: x + np.linspace(0, x.std(), n_points),
            'regime_shifts': lambda x: x + np.where(np.random.random(n_points) > 0.95, x.std()*2, 0)
        }),
        
        'multiple_seasonality': (
            50 + 20 * np.sin(x) + 10 * np.sin(7*x), {
            'perfect': lambda x: x,
            'missing_short_cycle': lambda x: 50 + 20 * np.sin(x),
            'amplitude_ratio_error': lambda x: 50 + 15 * np.sin(x) + 15 * np.sin(7*x),
            'noisy': lambda x: x + np.random.normal(0, 3, n_points)
        }),
        
        'trend_change': (
            np.concatenate([
                np.linspace(0, 50, n_points//2),
                np.linspace(50, 30, n_points-n_points//2)
            ]), {
            'perfect': lambda x: x,
            'missed_reversal': lambda x: np.concatenate([
                np.linspace(0, 50, n_points//2),
                np.linspace(50, 50, n_points-n_points//2)
            ]),
            'late_detection': lambda x: np.roll(x, n_points//10),
            'overreaction': lambda x: x * np.where(np.arange(n_points) > n_points//2, 0.7, 1.0)
        }),
        
        'cyclic_with_trend': (
            np.linspace(0, 100, n_points) + 20 * signal.sawtooth(x), {
            'perfect': lambda x: x,
            'trend_only': lambda x: np.linspace(0, 100, n_points),
            'cycle_only': lambda x: 50 + 20 * signal.sawtooth(x),
            'magnitude_error': lambda x: np.linspace(0, 100, n_points) + 10 * signal.sawtooth(x)
        }),
        
        'exponential_growth': (
            np.exp(np.linspace(0, 1, n_points)), {
            'perfect': lambda x: x,
            'linear_approximation': lambda x: np.linspace(1, np.max(x), n_points),
            'underestimate': lambda x: x**0.8,
            'delayed_response': lambda x: np.roll(x, n_points//20),
            'noisy': lambda x: x * (1 + np.random.normal(0, 0.05, n_points))
        })
    } 
    # Generate predictions and calculate metrics
    results = {}
    metrics = {}
    
    for pattern_name, (baseline, predictors) in patterns.items():
        #print(patterns.keys())
        results[f'{pattern_name}_baseline'] = baseline
        pattern_metrics = {}
        
        print(f"pattern_name: {pattern_name}")
        for pred_name, pred_func in predictors.items():
            
            prediction = pred_func(baseline)
            print(f"""
            pred_name : {pred_name},
            baseline : {baseline},
            prediction : {prediction}
            """)
            results[f'{pattern_name}_{pred_name}'] = prediction
            
            # Calculate metrics


            metrics_eval = Evaluate(target_values=baseline,prediction=prediction)
            
            pattern_metrics[pred_name] = {
                'MSE': mse_metric(baseline, prediction),
                'MAE': mae_metric(baseline, prediction),
                "RMSE" : rmse_metric(baseline, prediction),
                #"R2" : r2_score(baseline,prediction),
                "MAPE" : mape_metric(baseline,prediction),
                "WMAPE" : wmape_metric(baseline,prediction),
                "NRMSE" : nrmse_metric(baseline,prediction),
                "Overlay_area": metrics_eval.overlay_dx_area_under_curve_metric(
                    prediction,
                    100,
                    0,
                    0.1
                )}

        print(f"pattern over : {pattern_name}")
            

        metrics[pattern_name] = pattern_metrics
        print(pattern_metrics)
    
    return pd.DataFrame(results, index=date_rng), metrics

def generate_overlay_for_fake_cases(
    n_points=1000, freq="D",
    save_in_file=False,
    saving_path=None):
    """Generate baselines and corresponding predictions with controlled deviations."""
    date_rng = pd.date_range(start='2023-01-01', periods=n_points, freq=freq)
    x = np.linspace(0, 4*np.pi, n_points)

    patterns = {
        'constant': (np.ones(n_points) * 100, {
            'perfect': lambda x: x,
            'noisy': lambda x: x + np.random.normal(0, 5, n_points),
            'biased': lambda x: x * 1.1,
            'delayed': lambda x: np.roll(x, 5),
            'outliers': lambda x: x + np.where(np.random.random(n_points) > 0.98, 50, 0)
        }),
        
        'linear_trend': (np.linspace(0, 100, n_points), {
            'perfect': lambda x: x,
            'underestimate': lambda x: x * 0.8,
            'overestimate': lambda x: x * 1.2,
            'lagged': lambda x: np.roll(x, 10),
            'nonlinear_bias': lambda x: x + 0.001 * x**2,
            'step_changes': lambda x: x + np.where(x > 50, 20, 0)
        }),
        
        'seasonal': (50 + 30 * np.sin(x), {
            'perfect': lambda x: x,
            'amplitude_error': lambda x: 50 + 20 * np.sin(x),
            'phase_shift': lambda x: 50 + 30 * np.sin(x + 0.5),
            'missed_peaks': lambda x: np.where(x > 60, 60, x),
            'frequency_error': lambda x: 50 + 30 * np.sin(1.1 * x),
            'asymmetric_error': lambda x: x + np.where(np.sin(x) > 0, 5, -2)
        }),
        
        'random_walk': (np.cumsum(np.random.normal(0, 0.1, n_points)), {
            'perfect': lambda x: x,
            'smoothed': lambda x: pd.Series(x).rolling(5).mean().fillna(method='bfill'),
            'delayed': lambda x: np.roll(x, 3),
            'noisy': lambda x: x + np.random.normal(0, x.std()*0.1, n_points),
            'trend_biased': lambda x: x + np.linspace(0, x.std(), n_points),
            'regime_shifts': lambda x: x + np.where(np.random.random(n_points) > 0.95, x.std()*2, 0)
        }),
        
        'multiple_seasonality': (
            50 + 20 * np.sin(x) + 10 * np.sin(7*x), {
            'perfect': lambda x: x,
            'missing_short_cycle': lambda x: 50 + 20 * np.sin(x),
            'amplitude_ratio_error': lambda x: 50 + 15 * np.sin(x) + 15 * np.sin(7*x),
            'noisy': lambda x: x + np.random.normal(0, 3, n_points)
        }),
        
        'trend_change': (
            np.concatenate([
                np.linspace(0, 50, n_points//2),
                np.linspace(50, 30, n_points-n_points//2)
            ]), {
            'perfect': lambda x: x,
            'missed_reversal': lambda x: np.concatenate([
                np.linspace(0, 50, n_points//2),
                np.linspace(50, 50, n_points-n_points//2)
            ]),
            'late_detection': lambda x: np.roll(x, n_points//10),
            'overreaction': lambda x: x * np.where(np.arange(n_points) > n_points//2, 0.7, 1.0)
        }),
        
        'cyclic_with_trend': (
            np.linspace(0, 100, n_points) + 20 * signal.sawtooth(x), {
            'perfect': lambda x: x,
            'trend_only': lambda x: np.linspace(0, 100, n_points),
            'cycle_only': lambda x: 50 + 20 * signal.sawtooth(x),
            'magnitude_error': lambda x: np.linspace(0, 100, n_points) + 10 * signal.sawtooth(x)
        }),
        
        'exponential_growth': (
            np.exp(np.linspace(0, 1, n_points)), {
            'perfect': lambda x: x,
            'linear_approximation': lambda x: np.linspace(1, np.max(x), n_points),
            'underestimate': lambda x: x**0.8,
            'delayed_response': lambda x: np.roll(x, n_points//20),
            'noisy': lambda x: x * (1 + np.random.normal(0, 0.05, n_points))
        })
    }
    for pattern_name, (baseline, predictors) in patterns.items():
        #print(patterns.keys())
        results = {}
        results[f'{pattern_name}_baseline'] = baseline
        
        print(f"pattern_name: {pattern_name}")
        pattern_cases = {}
        metrics_eval = Evaluate(target_values=baseline,prediction=None)
        for pred_name, pred_func in predictors.items():
            
            prediction = pred_func(baseline)
            print(f"""
            pred_name : {pred_name},
            baseline : {baseline},
            prediction : {prediction}
            """)
            results[f'{pattern_name}_{pred_name}'] = prediction
            
        df_overlay = pd.DataFrame(results)
        metrics_eval.overlay_dx_visualisation_df(
        forecasts_df=df_overlay,max_percentage=100,min_percentage=0,step=0.1,
        save_in_file=save_in_file,
        saving_path=os.path.join(saving_path,f"{pattern_name}"),
        )
        plt.close()
    


def generate_pred_plots_for_fake_cases(
    n_points=1000, freq="D",
    save_in_file=False,
    saving_path=None):
    """Generate baselines and corresponding predictions with controlled deviations."""
    date_rng = pd.date_range(start='2023-01-01', periods=n_points, freq=freq)
    x = np.linspace(0, 4*np.pi, n_points)

    patterns = {
        'constant': (np.ones(n_points) * 100, {
            'perfect': lambda x: x,
            'noisy': lambda x: x + np.random.normal(0, 5, n_points),
            'biased': lambda x: x * 1.1,
            'delayed': lambda x: np.roll(x, 5),
            'outliers': lambda x: x + np.where(np.random.random(n_points) > 0.98, 50, 0)
        }),
        
        'linear_trend': (np.linspace(0, 100, n_points), {
            'perfect': lambda x: x,
            'underestimate': lambda x: x * 0.8,
            'overestimate': lambda x: x * 1.2,
            'lagged': lambda x: np.roll(x, 10),
            'nonlinear_bias': lambda x: x + 0.001 * x**2,
            'step_changes': lambda x: x + np.where(x > 50, 20, 0)
        }),
        
        'seasonal': (50 + 30 * np.sin(x), {
            'perfect': lambda x: x,
            'amplitude_error': lambda x: 50 + 20 * np.sin(x),
            'phase_shift': lambda x: 50 + 30 * np.sin(x + 0.5),
            'missed_peaks': lambda x: np.where(x > 60, 60, x),
            'frequency_error': lambda x: 50 + 30 * np.sin(1.1 * x),
            'asymmetric_error': lambda x: x + np.where(np.sin(x) > 0, 5, -2)
        }),
        
        'random_walk': (np.cumsum(np.random.normal(0, 0.1, n_points)), {
            'perfect': lambda x: x,
            'smoothed': lambda x: pd.Series(x).rolling(5).mean().fillna(method='bfill'),
            'delayed': lambda x: np.roll(x, 3),
            'noisy': lambda x: x + np.random.normal(0, x.std()*0.1, n_points),
            'trend_biased': lambda x: x + np.linspace(0, x.std(), n_points),
            'regime_shifts': lambda x: x + np.where(np.random.random(n_points) > 0.95, x.std()*2, 0)
        }),
        
        'multiple_seasonality': (
            50 + 20 * np.sin(x) + 10 * np.sin(7*x), {
            'perfect': lambda x: x,
            'missing_short_cycle': lambda x: 50 + 20 * np.sin(x),
            'amplitude_ratio_error': lambda x: 50 + 15 * np.sin(x) + 15 * np.sin(7*x),
            'noisy': lambda x: x + np.random.normal(0, 3, n_points)
        }),
        
        'trend_change': (
            np.concatenate([
                np.linspace(0, 50, n_points//2),
                np.linspace(50, 30, n_points-n_points//2)
            ]), {
            'perfect': lambda x: x,
            'missed_reversal': lambda x: np.concatenate([
                np.linspace(0, 50, n_points//2),
                np.linspace(50, 50, n_points-n_points//2)
            ]),
            'late_detection': lambda x: np.roll(x, n_points//10),
            'overreaction': lambda x: x * np.where(np.arange(n_points) > n_points//2, 0.7, 1.0)
        }),
        
        'cyclic_with_trend': (
            np.linspace(0, 100, n_points) + 20 * signal.sawtooth(x), {
            'perfect': lambda x: x,
            'trend_only': lambda x: np.linspace(0, 100, n_points),
            'cycle_only': lambda x: 50 + 20 * signal.sawtooth(x),
            'magnitude_error': lambda x: np.linspace(0, 100, n_points) + 10 * signal.sawtooth(x)
        }),
        
        'exponential_growth': (
            np.exp(np.linspace(0, 1, n_points)), {
            'perfect': lambda x: x,
            'linear_approximation': lambda x: np.linspace(1, np.max(x), n_points),
            'underestimate': lambda x: x**0.8,
            'delayed_response': lambda x: np.roll(x, n_points//20),
            'noisy': lambda x: x * (1 + np.random.normal(0, 0.05, n_points))
        })
    }
    for pattern_name, (baseline, predictors) in patterns.items():
        #print(patterns.keys())
        results = {}
        results[f'{pattern_name}_baseline'] = baseline
        
        print(f"pattern_name: {pattern_name}")
        for pred_name, pred_func in predictors.items():
            
            prediction = pred_func(baseline)
            print(f"""
            pred_name : {pred_name},
            baseline : {baseline},
            prediction : {prediction}
            """)
            results[f'{pattern_name}_{pred_name}'] = prediction

        df = pd.DataFrame(results)
        plt.figure(figsize=(10, 6))
        for column in df.columns:
            plt.plot(date_rng, df[column], label=column)
        plt.title(f"plot for case {pattern_name}")
        plt.xlabel("Time")
        plt.ylabel("Values")
        plt.legend()
        plt.savefig(os.path.join(saving_path,f"fake_data_{pattern_name}_{pred_name}"))
        plt.show()
        plt.close()






def generate_fake_patterns_for_heatmap(
    n_points=1000, freq="D",
    ):
    date_rng = pd.date_range(start='2023-01-01', periods=n_points, freq=freq)
    x = np.linspace(0, 4*np.pi, n_points)

    patterns = {
        'constant': (np.ones(n_points) * 100, {
            'perfect': lambda x: x,
            'noisy': lambda x: x + np.random.normal(0, 5, n_points),
            'biased': lambda x: x * 1.1,
            'delayed': lambda x: np.roll(x, 5),
            'outliers': lambda x: x + np.where(np.random.random(n_points) > 0.98, 50, 0)
        }),
        
        'linear_trend': (np.linspace(0, 100, n_points), {
            'perfect': lambda x: x,
            'underestimate': lambda x: x * 0.8,
            'overestimate': lambda x: x * 1.2,
            'lagged': lambda x: np.roll(x, 10),
            'nonlinear_bias': lambda x: x + 0.001 * x**2,
            'step_changes': lambda x: x + np.where(x > 50, 20, 0)
        }),
        
        'seasonal': (50 + 30 * np.sin(x), {
            'perfect': lambda x: x,
            'amplitude_error': lambda x: 50 + 20 * np.sin(x),
            'phase_shift': lambda x: 50 + 30 * np.sin(x + 0.5),
            'missed_peaks': lambda x: np.where(x > 60, 60, x),
            'frequency_error': lambda x: 50 + 30 * np.sin(1.1 * x),
            'asymmetric_error': lambda x: x + np.where(np.sin(x) > 0, 5, -2)
        }),
        
        'random_walk': (np.cumsum(np.random.normal(0, 0.1, n_points)), {
            'perfect': lambda x: x,
            'smoothed': lambda x: pd.Series(x).rolling(5).mean().fillna(method='bfill'),
            'delayed': lambda x: np.roll(x, 3),
            'noisy': lambda x: x + np.random.normal(0, x.std()*0.1, n_points),
            'trend_biased': lambda x: x + np.linspace(0, x.std(), n_points),
            'regime_shifts': lambda x: x + np.where(np.random.random(n_points) > 0.95, x.std()*2, 0)
        }),
        
        'multiple_seasonality': (
            50 + 20 * np.sin(x) + 10 * np.sin(7*x), {
            'perfect': lambda x: x,
            'missing_short_cycle': lambda x: 50 + 20 * np.sin(x),
            'amplitude_ratio_error': lambda x: 50 + 15 * np.sin(x) + 15 * np.sin(7*x),
            'noisy': lambda x: x + np.random.normal(0, 3, n_points)
        }),
        
        'trend_change': (
            np.concatenate([
                np.linspace(0, 50, n_points//2),
                np.linspace(50, 30, n_points-n_points//2)
            ]), {
            'perfect': lambda x: x,
            'missed_reversal': lambda x: np.concatenate([
                np.linspace(0, 50, n_points//2),
                np.linspace(50, 50, n_points-n_points//2)
            ]),
            'late_detection': lambda x: np.roll(x, n_points//10),
            'overreaction': lambda x: x * np.where(np.arange(n_points) > n_points//2, 0.7, 1.0)
        }),
        
        'cyclic_with_trend': (
            np.linspace(0, 100, n_points) + 20 * signal.sawtooth(x), {
            'perfect': lambda x: x,
            'trend_only': lambda x: np.linspace(0, 100, n_points),
            'cycle_only': lambda x: 50 + 20 * signal.sawtooth(x),
            'magnitude_error': lambda x: np.linspace(0, 100, n_points) + 10 * signal.sawtooth(x)
        }),
        
        'exponential_growth': (
            np.exp(np.linspace(0, 1, n_points)), {
            'perfect': lambda x: x,
            'linear_approximation': lambda x: np.linspace(1, np.max(x), n_points),
            'underestimate': lambda x: x**0.8,
            'delayed_response': lambda x: np.roll(x, n_points//20),
            'noisy': lambda x: x * (1 + np.random.normal(0, 0.05, n_points))
        })
    }
    # generate all those cases and put those time series in a dict
    results ={}
    curve_name_to_pattern_name = {}
    for pattern_name, (baseline, predictors) in patterns.items():
        #print(patterns.keys())
        for case_name,case_func in predictors.items():
            results[f'{pattern_name}_{case_name}'] = case_func(baseline)
            curve_name_to_pattern_name[f'{pattern_name}_{case_name}'] = pattern_name

    return date_rng,results ,curve_name_to_pattern_name



def measure_metrics_and_plot_heatmap(
    date_rng,
    results,
    curve_name_to_pattern_name,
    save_in_file=False,
    saving_path=None
):
    metrics_res = {}
    metric_list = {
    rmse_metric :"rmse",
    mse_metric:"mse",
    mae_metric:"mae",
    mape_metric:"mape",
    wmape_metric:"wmape",
    nrmse_metric:"nrmse",
    overlay_metric:"overlay",
    }
    for metric in metric_list:
        metric_res = {}
        for case1 in tqdm(results):
            for case2 in results:
                pattern_name1 = curve_name_to_pattern_name[case1]
                pattern_name2 = curve_name_to_pattern_name[case2]
                if pattern_name1 == pattern_name2:
                    print(f"pattern_name1 : {pattern_name1} , pattern_name2 : {pattern_name2}")
                    continue
                f1 = results[case1]
                f2 = results[case2]
                metric_res[f"{case1}_{case2}"] = metric(f1,f2)
        metrics_res[metric_list[metric]] = cp.deepcopy(metric_res)
    
    return metrics_res
