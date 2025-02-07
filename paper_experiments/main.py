import glob
import os

import catboost
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from generate_fake_patterns import (
    generate_baseline_and_predictions,
    generate_overlay_for_fake_cases,
    generate_pred_plots_for_fake_cases,
)
from sklearn.model_selection import TimeSeriesSplit
from train import allRegressors
from utils import correlation_heatmap, overlay_plot_all_models, prediction_graphs


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
    #title="Correlation heatmap between metrics on all the trained models on the Beijin pollution dataset",
    save_in_file=True,
    save_path=os.path.join(saving_path,"heatmap_run3_chinese_pollution_all_metrics.png")
    )

    predictions = [r[2] for r in res]
    prediction_graphs(
        prediction=predictions[2]["overlay_metric"],
        downscaling_factor=1000,
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
    metric_names = ["MSE","MAE","RMSE","MAPE",
    #"MAPE","SMAPE",
    "WMAPE","NRMSE","Overlay_area"]
    dic = {
        metric_name : [] for metric_name in metric_names
    }
    for case in metrics:
        for subcase in metrics[case]:
            print(subcase)
            for metric_name in metric_names:
                dic[metric_name].append(
                    metrics[case][subcase][metric_name]
                )

    df = pd.DataFrame(dic)
        #df.set_index(pd.Index(cases))
    saving_path = "plots/fake_data"
    #print(df)
    #import pdb; pdb.set_trace()
    df.to_csv(os.path.join(saving_path,"metric_results_fake_data.csv"))
    df.corr().to_csv(os.path.join(saving_path,"corr_matrix_results.csv"))
    correlation_heatmap(df_metrics=df,
    #title="Correlation heatmap between metrics on all the trained models on the  generated dataset",
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

def chinese_ETT():
    path = "/home/ubuntu/amine/forecaster/paper_experiments/datasets/ETDataset/ETT-small/ETTm2.csv"
    df = pd.read_csv(path)
    cols = df.columns.to_list()
    print(cols)
    cols.remove("date")
    print(cols)
    data = df[cols].copy()
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits)
    res = []
    x_cols = cols[:]
    x_cols.remove("OT")
    X = df[x_cols]
    y = df["OT"]
    res = []

    saving_path = "plots/chinese_ETT"
    for i, (train_index, test_index) in enumerate(tscv.split(data)):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
        print(X_test.shape)
        res.append(allRegressors(X_train, X_test, y_train, y_test,run_name=f"ETT_data_run_{i}",saving_path=saving_path))
    dfs = [r[0] for r in res]
    df_metrics = dfs[2]
    correlation_heatmap(df_metrics=df_metrics,
    #title="Correlation heatmap between metrics on all the trained models on the chinese ET dataset",
    save_in_file=True,
    save_path=os.path.join(saving_path,"heatmap_run3_chinese_ETT_all_metrics.png")
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

def Electricity_load():
    data = pd.read_csv("/home/ubuntu/amine/forecaster/paper_experiments/datasets/LD2011_2014.txt",sep=";")
    data.reset_index(drop=True)
    df = data
    df = df.drop(df.columns[0],axis =1)
    df = df.apply(lambda x: x.str.replace(',', '.', regex=True).str.strip() if x.dtype == 'object' else x)
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert all to float, replacing errors with NaN
    df.fillna(0, inplace=True)  # Replace NaNs with 0 (or another default value)
    data = df

    X = data.drop(data.columns[-1],axis=1).drop(data.columns[0],axis=1)
    y = data["MT_001"]
    #X = X.values
    #y = y.values
    n_splits = 3
    tscv = TimeSeriesSplit(n_splits=n_splits)

    res = []
    saving_path = "plots/electricity_load_uc_irvine"
    prefix = "Electricity_load_uc_irvine"
    for i, (train_index, test_index) in enumerate(tscv.split(data)):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        X_test = X.iloc[test_index]
        y_test = y.iloc[test_index]
        X_train = X_train.values
        X_test = X_test.values
        y_train = y_train.values
        y_test = y_test.values
        print(X_test.shape)
        try:
            res.append((allRegressors(X_train, X_test, y_train, y_test,run_name=f"{prefix}_data_run_{i}",saving_path=saving_path)))
        except catboost.CatBoostError:
            res.append([])
    #return res
    dfs = [r[0] for r in res]
    df_metrics = dfs[2]
    correlation_heatmap(df_metrics=df_metrics,
    #title="Correlation heatmap between metrics on all the trained models on the Beijin pollution dataset",
    save_in_file=True,
    save_path=os.path.join(saving_path,"heatmap_run3_{prefix}_all_metrics.png")
    )

    predictions = [r[2] for r in res]
    prediction_graphs(
        prediction=predictions[2]["overlay_metric"],
        downscaling_factor=1000,
        save_in_file=True,
        save_path=saving_path)

    overlay_plot_all_models(
        prediction=predictions[2],
        save_in_file=True,
        saving_path=saving_path)


if __name__ == "__main__":
    Electricity_load()
    chinese_ETT()
    fake_datasets()
    res = chinese_weather()