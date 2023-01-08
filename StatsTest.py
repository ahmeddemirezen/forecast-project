from statsforecast.models import AutoARIMA, AutoETS, Naive, AutoCES, AutoTheta
from statsforecast import StatsForecast
import matplotlib.pyplot as plt
import Forecasts as fcs
import pandas as pd
import numpy as np
import time as t
time = t.time()


def MAD(forecast, actual):
    return np.mean(np.abs(forecast - actual))


def MSE(forecast, actual):
    return np.mean((forecast - actual)**2)


def MAPE(forecast, actual):
    return np.mean(np.abs((forecast - actual)/actual))


def AIForecast(data, product):
    df = pd.DataFrame({'y': data[str(product)]})
    df['unique_id'] = 1
    df['ds'] = pd.date_range(start='2018-10-01', periods=len(df), freq='W')
    Y_train_df = df.iloc[:len(df)-4]  # 7 is the number of weeks to forecast
    Y_test_df = df.iloc[len(df)-4:]  # 7 is the number of weeks to forecast

    # print(Y_train_df)
    # print(Y_test_df)

    season_length = 13
    horizon = len(Y_test_df)

    models = [
        AutoARIMA(season_length=season_length),
        AutoETS(season_length=season_length),
        AutoCES(season_length=season_length),
        AutoTheta(season_length=season_length)
    ]

    sf = StatsForecast(
        df=Y_train_df,
        models=models,
        freq='W',
        n_jobs=-1,
    )

    fc = fcs.Forecast(df['y'].values)

    Y_hat_df = sf.forecast(horizon)
    es = fc.OptimalES()
    ma = fc.OptimalMA(maxStep=13)
    lr = fc.LinearRegression()

    Y_hat_df['ES'] = es.result[-horizon:]
    Y_hat_df['MA'] = ma.result[-horizon:]
    Y_hat_df['LR'] = lr.result[-horizon:]

    Y_hat_df.reset_index()

    Y_hat_df = Y_test_df.merge(Y_hat_df, how='left', on=['unique_id', 'ds'])

    fig, ax = plt.subplots(1, 1, figsize=(20, 7))
    plot_df = pd.concat([Y_train_df, Y_hat_df]).set_index('ds')
    plot_df[['y', 'AutoARIMA', 'AutoETS', 'CES', 'AutoTheta',
             'ES', 'MA', 'LR']].plot(ax=ax, linewidth=2)
    ax.set_title('Forecasting', fontsize=20)
    ax.set_ylabel('y', fontsize=16)
    ax.set_xlabel('ds', fontsize=16)
    ax.legend(prop={'size': 16})
    ax.grid()

    plt.savefig(str(product)+'forecast.png')

    y_true = Y_hat_df['y'].values

    ets_pred = Y_hat_df['AutoETS'].values
    arima_pred = Y_hat_df['AutoARIMA'].values
    ces_pred = Y_hat_df['CES'].values
    theta_pred = Y_hat_df['AutoTheta'].values

    result_df = pd.DataFrame()
    result_df = Y_hat_df
    error_df = pd.DataFrame()
    error_df['Methods'] = ['ETS', 'ARIMA', 'CES', 'Theta', 'ES ' +
                           str(es.params), 'MA' + str(ma.params), 'LR ' + str(lr.params)]
    error_df['MAD'] = [MAD(ets_pred, y_true), MAD(arima_pred, y_true), MAD(ces_pred, y_true), MAD(theta_pred, y_true), MAD(
        es.result[-horizon:], y_true), MAD(ma.result[-horizon:], y_true), MAD(lr.result[-horizon:], y_true)]
    error_df['MSE'] = [MSE(ets_pred, y_true), MSE(arima_pred, y_true), MSE(ces_pred, y_true), MSE(theta_pred, y_true), MSE(
        es.result[-horizon:], y_true), MSE(ma.result[-horizon:], y_true), MSE(lr.result[-horizon:], y_true)]
    error_df['MAPE'] = [MAPE(ets_pred, y_true), MAPE(arima_pred, y_true), MAPE(ces_pred, y_true), MAPE(theta_pred, y_true), MAPE(
        es.result[-horizon:], y_true), MAPE(ma.result[-horizon:], y_true), MAPE(lr.result[-horizon:], y_true)]

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter(str(product)+'.xlsx', engine='xlsxwriter')

    # Write each dataframe to a different worksheet.
    result_df.to_excel(writer, sheet_name='Results')
    error_df.to_excel(writer, sheet_name='Errors')

    # Close the Pandas Excel writer and output the Excel file.
    writer.close()


# import data
data = pd.read_excel('data.xlsx', sheet_name='Sheet1')
print(data.head())

time = t.time()
print('Start')
columns = ["p1", "p2", "p3",  "p4",  "p5",  "p6",  "p7",  "p8",  "p9",  "p10", "p11",  "p12",  "p13",  "p14",  "p15",  "p16",  "p17",
           "p18",  "p19",  "p20",  "p21",  "p22",  "p23",  "p24",  "p25",  "p26",  "p27",  "p28",  "p29",  "p30",  "p31",  "p32",  "p33"]
for i in columns:
    if (i == 'id' or i == 'date'):
        continue
    AIForecast(data, i)
    print(i+' done in ', t.time() - time, ' seconds')
    time = t.time()
