from generate_fake_patterns import (
    generate_baseline_and_predictions ,
    generate_overlay_for_fake_cases,
    generate_pred_plots_for_fake_cases
)
import os
from grid_search import GPURegressorGridSearch
import pandas as pd 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import glob
import statistics

# ignore warnings
import warnings
from train import allRegressors
from utils import (
    prediction_graphs,
    correlation_heatmap,
    overlay_plot_all_models
)
def chinese_weather():
    # read data
    filePath = '/home/ubuntu/amine/forecaster/paper_experiments/datasets/beijing-multisite-airquality-data-set'
    allFiles = glob.glob(filePath + "/*.csv")
    dataFrames = []
    for i in allFiles:
        df = pd.read_csv(i, index_col=None, header=0)
        dataFrames.append(df)
    data = pd.concat(dataFrames)


    # preprocessing 
    data.drop(["No"], axis=1, inplace=True)
    data.rename(columns = {'year': 'Year',
                           'month': 'Month',
                           'day': "Day",
                           'hour': 'Hour',
                           'pm2.5': 'PM2.5',
                           'DEWP': 'DewP',
                           'TEMP': 'Temp',
                           'PRES': 'Press',
                           'RAIN': 'Rain',
                           'wd': 'WinDir',
                           'WSPM': 'WinSpeed',
                           'station': 'Station'}, inplace = True)



    # fill the null values in numerical columns with average specific to certain column
    # fill in the missing data in the columns according to the Month average.
    unique_Month = pd.unique(data.Month)

    # find PM2_5 averages in Month specific
    # Equalize the average PM2.5 values to the missing values in PM2_5 specific to Month
    temp_data = data.copy()  # set temp_data variable to avoid losing real data
    columns = ["PM2.5", 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'Temp', 'Press', 'DewP', 'Rain', 'WinSpeed'] # it can be add more column
    for c in unique_Month:

        # create Month filter
        Month_filtre = temp_data.Month == c
        # filter data by Month
        fitered_data = temp_data[Month_filtre]

        # find average for PM2_5 in specific to Month
        for s in columns:
            mean = np.round(np.mean(fitered_data[s]), 2)
            if ~np.isnan(mean): # if there if average specific to Month
                fitered_data[s] = fitered_data[s].fillna(mean)
                print(f"Missing Value in {s} column fill with {mean} when Month:{c}")
            else: # find average for all data if no average in specific to Month
                all_data_mean = np.round(np.mean(data[s]),2)
                fitered_data[s] = fitered_data[s].fillna(all_data_mean)
                print(f"Missing Value in {s} column fill with {all_data_mean}")
        # Synchronize data filled with missing values in PM2.5 to data temporary            
        temp_data[Month_filtre] = fitered_data

    # equate the deprecated temporary data to the real data variable
    data = temp_data.copy() 



    data['Date']=pd.to_datetime(data[['Year', 'Month', 'Day']])

    # function to find day of the week based on the date field
    import calendar
    def findDay(date): 
        dayname = calendar.day_name[date.weekday()]
        return dayname
    data['DayNames'] = data['Date'].apply(lambda x: findDay(x))


    data.drop(["DayNames", "Date", "PM10", "Year", "Month", "Day", "Hour"], axis=1, inplace=True)


    from sklearn.preprocessing import LabelEncoder
    # define a function for label encoding
    def labelEncoder(labelColumn):
        labelValues = labelColumn
        unique_labels = labelColumn.unique()
        le = LabelEncoder()
        labelColumn = le.fit_transform(labelColumn)
        print('Encoding Approach:')
        for i, j in zip(unique_labels, labelColumn[np.sort(np.unique(labelColumn, return_index=True)[1])]): 
            print(f'{i}  ==>  {j}')
        return labelColumn

    categorical_variables = ["WinDir", "Station"]
    for i in categorical_variables:
        print(f"For {i} column ")
        data[f"{i}"] = labelEncoder(data[f"{i}"])
        print("**********************************")


    # create input and output
    X = data.drop('PM2.5', axis = 1)
    y = data['PM2.5']

    from sklearn.model_selection import TimeSeriesSplit

    n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits)

    res = []
    saving_path = "plots/chinese_weather"
    for i, (train_index, test_index) in enumerate(tscv.split(data)):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
        print(X_test.shape)
        res.append(allRegressors(X_train, X_test, y_train, y_test,run_name=f"weather_data_run_{i}",saving_path=saving_path))
    #return res
    dfs = [r[0] for r in res]
    df_metrics = dfs[2]
    correlation_heatmap(df_metrics=df_metrics,
    title="Correlation heatmap between metrics on all the trained models on the Beijin pollution dataset",
    save_in_file=True,
    save_path=os.path.join(saving_path,"heatmap_run3_chinese_pollution_all_metrics.png")
    )

    predictions = [r[2] for r in res]
    prediction_graphs(
        prediction=predictions[2]["overlay_metric"],
        downscaling_factor=500,
        save_in_file=True,
        save_path=saving_path)

    overlay_plot_all_models(
        prediction=predictions[2],
        save_in_file=True,
        saving_path=saving_path)

def fake_datasets():
    df, metrics = generate_baseline_and_predictions()
    cases = []
    for case in metrics:
        for subcase in metrics[case]:
            cases.append(
            case+"_"+f"{subcase}"
            )
    #df.set_index(pd.Index(cases))
    saving_path = "plots/fake_data"
    #print(df)
    #import pdb; pdb.set_trace()
    correlation_heatmap(df_metrics=df,
    title="Correlation heatmap between metrics on all the trained models on the  generated dataset",
    save_in_file=True,
    save_path=os.path.join(saving_path,"heatmap_metrics_generated_dataset.png")
    )
    # overlay
    generate_overlay_for_fake_cases(
        n_points=1000,
        freq="D",
        save_in_file=True,
        saving_path=saving_path
    )
    #generate pred plots
    generate_pred_plots_for_fake_cases(
        save_in_file=True,
        saving_path=saving_path
    )


if __name__ == "__main__":
    fake_datasets()
    #res = main()