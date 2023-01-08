import Forecasts as fc
import pandas as pd

data = pd.read_excel('data.xlsx', sheet_name='Sheet1')

horizon = 4

# print(len(nonSeasonal), len(seasonal))

df = pd.DataFrame({'y': data['p6']})
df['unique_id'] = 1
df['ds'] = pd.date_range(start='2018-10-01', periods=len(df), freq='W')

fc = fc.Forecast(df['y'].values)

fc.OptimalMA().Print()

# es = fc.OptimalES().result[-horizon:]
# ma = fc.MovingAverage(3).Print()
# ma = fc.MovingAverage(4).Print()
# ma = fc.MovingAverage(5).Print()
# ma = fc.MovingAverage(6).Print()
# ma = fc.MovingAverage(7).Print()
# ma = fc.MovingAverage(13).Print()
# lr = fc.LinearRegression().result[-horizon:]



# print(es)
# print(ma)
# print(lr)
