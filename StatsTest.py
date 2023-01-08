import time as t
time = t.time()
import numpy as np
import pandas as pd
import Forecasts as fc

import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, Naive, AutoCES, AutoTheta

print('libraries imported with in:', t.time() - time, 'seconds')
time = t.time()

def MAD(forecast, actual):
    return np.mean(np.abs(forecast - actual))
def MSE(forecast, actual):
    return np.mean((forecast - actual)**2)
def MAPE(forecast, actual):
    return np.mean(np.abs((forecast - actual)/actual))

# import data
data = pd.read_excel('data.xlsx', sheet_name='Sheet1')
print(data.head())


# print(len(nonSeasonal), len(seasonal))

df = pd.DataFrame({'y': data['p1']})
df['unique_id'] = 1
df['ds'] = data['date']
Y_train_df = df.iloc[:len(df)-4] # 4 is the number of weeks to forecast
Y_test_df = df.iloc[len(df)-4:] # 4 is the number of weeks to forecast

# print(Y_train_df)
# print(Y_test_df)

season_length = 13
horizon = len(Y_test_df)

print('data imported with in:', t.time() - time, 'seconds')
time = t.time()

models = [
    AutoARIMA(season_length=season_length),
    AutoETS(season_length=season_length),
    AutoCES(season_length=season_length),
    AutoTheta(season_length=season_length)
]

sf = StatsForecast(
    df = Y_train_df,
    models = models,
    freq='W',
    n_jobs=-1,
)

fc = fc.Forecast(df['y'].values)


print('models imported with in:', t.time() - time, 'seconds')
time = t.time()

Y_hat_df = sf.forecast(horizon)
es = fc.OptimalES()
ma = fc.MovingAverage(3)
lr = fc.LinearRegression()

print('forecasts imported with in:', t.time() - time, 'seconds')
time = t.time()

Y_hat_df['ES'] = es.result[-horizon:]
Y_hat_df['MA'] = ma.result[-horizon:]
Y_hat_df['LR'] = lr.result[-horizon:]

Y_hat_df.reset_index()
print(Y_hat_df.head())

Y_hat_df = Y_test_df.merge(Y_hat_df, how='left', on = ['unique_id', 'ds'])

fig, ax = plt.subplots(1, 1, figsize=(20, 7))
plot_df = pd.concat([Y_train_df, Y_hat_df]).set_index('ds')
plot_df[['y', 'AutoARIMA', 'AutoETS', 'CES', 'AutoTheta', 'ES', 'MA', 'LR']].plot(ax=ax, linewidth=2)
ax.set_title('Forecasting', fontsize=20)
ax.set_ylabel('y', fontsize=16)
ax.set_xlabel('ds', fontsize=16)
ax.legend(prop={'size': 16})
ax.grid()

print('plots imported with in:', t.time() - time, 'seconds')
time = t.time()

plt.show()

y_true = Y_hat_df['y'].values

ets_pred = Y_hat_df['AutoETS'].values
arima_pred = Y_hat_df['AutoARIMA'].values
ces_pred = Y_hat_df['CES'].values
theta_pred = Y_hat_df['AutoTheta'].values



print('ETS MAD: ', MAD(ets_pred, y_true))
# print('ETS MSE: ', MSE(ets_pred, y_true))
# print('ETS MAPE: ', MAPE(ets_pred, y_true))

print('ARIMA MAD: ', MAD(arima_pred, y_true))
# print('ARIMA MSE: ', MSE(arima_pred, y_true))
# print('ARIMA MAPE: ', MAPE(arima_pred, y_true))

print('CES MAD: ', MAD(ces_pred, y_true))

print('Theta MAD: ', MAD(theta_pred, y_true))

print('ES MAD: ', MAD(es.result[-horizon:], y_true))
print('MA MAD: ', MAD(ma.result[-horizon:], y_true))
print('LR MAD: ', MAD(lr.result[-horizon:], y_true))

print('MAD imported with in:', t.time() - time, 'seconds')








