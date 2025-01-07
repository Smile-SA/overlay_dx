import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
import sys
sys.path.append(os.path.abspath('..'))
from src.processing_forecasts.metrics import Evaluate






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
    
    # Base patterns
    #patterns = {
    #    'constant': (np.ones(n_points) * 100, {
    #        'perfect': lambda x: x,
    #        'noisy': lambda x: x + np.random.normal(0, 5, n_points),
    #        'biased': lambda x: x * 1.1,
    #        'delayed': lambda x: np.roll(x, 5)
    #    }),
    #    'linear_trend': (np.linspace(0, 100, n_points), {
    #        'perfect': lambda x: x,
    #        'underestimate': lambda x: x * 0.8,
    #        'overestimate': lambda x: x * 1.2,
    #        'lagged': lambda x: np.roll(x, 10)
    #    }),
    #    'seasonal': (50 + 30 * np.sin(x), {
    #        'perfect': lambda x: x,
    #        'amplitude_error': lambda x: 50 + 20 * np.sin(x),
    #        'phase_shift': lambda x: 50 + 30 * np.sin(x + 0.5),
    #        'missed_peaks': lambda x: np.where(x > 60, 60, x)
    #    }),
    #    'random_walk': (np.cumsum(np.random.normal(0, 0.1, n_points)), {
    #        'perfect': lambda x: x,
    #        'smoothe': lambda x: pd.Series(x).rolling(5).mean().fillna(method='bfill'),
    #        'delayed': lambda x: np.roll(x, 3),
    #        'noisy': lambda x: x + np.random.normal(0, x.std()*0.1, n_points)
    #    })
    #}
    #
    patterns = {
        #'constant': (np.ones(n_points) * 100, {
        #    'perfect': lambda x: x,
        #    'noisy': lambda x: x + np.random.normal(0, 5, n_points),
        #    'biased': lambda x: x * 1.1,
        #    'delayed': lambda x: np.roll(x, 5),
        #    'outliers': lambda x: x + np.where(np.random.random(n_points) > 0.98, 50, 0),
        #    'missing_values': lambda x: np.where(np.random.random(n_points) > 0.95, np.nan, x)
        #}),
        
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
            results[f'{pattern_name}_{pred_name}'] = prediction
            
            # Calculate metrics
            metrics_eval = Evaluate(target_values=baseline,prediction=prediction)
            
            pattern_metrics[pred_name] = {
                'MSE': mean_squared_error(baseline, prediction),
                'MAE': mean_absolute_error(baseline, prediction),
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

