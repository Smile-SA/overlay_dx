
import time
import uuid
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(".."))
# Decorator to measure execution time of each function call
def measure_time(execution_times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate a unique identifier for this call
            call_id = str(uuid.uuid4())  # Unique call ID

            start_time = time.time()  # Record start time
            result = func(*args, **kwargs)  # Execute the function
            end_time = time.time()  # Record end time

            # Calculate execution time
            execution_time = end_time - start_time

            # Store the execution time with the unique call ID in the provided dictionary
            metric_name = func.__name__
            if func.__name__ not in execution_times:
                execution_times[metric_name] = []

            execution_times[metric_name].append(execution_time)
            #print(execution_time.values())
            return result
        return wrapper
    return decorator


def correlation_heatmap(
    df_metrics: pd.DataFrame,
    title: str = None,
    save_in_file:bool = False,
    save_path: str|None = None,
):
    corr_matrix = df_metrics.corr()
    sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    )
    if title is None:
        plt.tight_layout()
    else:
        plt.title(title)
        plt.tight_layout()
    # saving part
    if save_in_file:
        plt.savefig(save_path) 
    plt.show()
    plt.close()



def prediction_graphs(
    prediction: dict,
    downscaling_factor: int | None = None,
    save_in_file:bool = False,
    save_path: str| None = None,
    ):
    "predicition here is the third element for one run  returned by AllRegressors in train.py , it is a dict containig for all regressors the y_true and y_pred "
    for regressor_name in prediction:
        test = prediction[regressor_name]["y_true"]
        y_pred = prediction[regressor_name]["y_pred"]
        dates = np.linspace(0,100,len(test))
        downsample_factor = downscaling_factor
        downsampled_dates = dates[::downsample_factor]
        downsampled_test = test[::downsample_factor]
        downsampled_y_pred = y_pred[::downsample_factor]
        dates = np.linspace(0,100,len(test))

   #     Create the plot
        plt.figure(figsize=(12, 6))
        confidence_interval = 1.96*np.std(test)
        plt.plot(downsampled_dates, downsampled_test, label='Test', color='blue', linewidth=2)
        plt.fill_between(downsampled_dates, downsampled_test - confidence_interval,downsampled_test + confidence_interval,color='lightblue', alpha=0.5, label='95% Confidence Interval')
        plt.plot(downsampled_dates, downsampled_y_pred, label='Prediction', color='orange', linestyle='--', linewidth=2)

        # Add labels, legend, and title
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.title(f'Test vs Prediction Time Series for model {regressor_name}', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True)

        # Show the plot
        plt.tight_layout()
        if save_in_file:
            plt.savefig(os.path.join(save_path,f"{regressor_name}_prediction_graph.png"))
        plt.show()

        plt.close()

def overlay_plot_all_models(
    prediction,
    save_in_file: bool = False,
    saving_path=None):
    from src.processing_forecasts.metrics import Evaluate
    df_overlay = {}
    prediction = prediction["rmse"]
    df_overlay["baseline"] = prediction["LinearRegression"]["y_true"]
    print(df_overlay["baseline"].shape)
    print(prediction["LinearRegression"]["y_pred"].shape)
    for model in prediction:
        df_overlay[model] = prediction[model]["y_pred"]

    df_overlay = pd.DataFrame(df_overlay)
    print(df_overlay)
    metric_exp = Evaluate(target_values=list(df_overlay["baseline"]),prediction=None).overlay_dx_visualisation_df(
        forecasts_df=df_overlay,max_percentage=100,min_percentage=0,step=0.1,
        save_in_file=save_in_file,
        saving_path=os.path.join(saving_path,"chinese_weather_overlay.png"),
    )
    plt.close()
