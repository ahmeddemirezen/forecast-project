import Forecasts as fc
import pandas as pd

nonSeasonal = [74,72,78,69,74,78,74,71,76,73,77,74,79,63,61,64,59,62,64,59,61,63,60,59,57,63,53,55,59,52,57,51,49,63,58,55,53,53,61]
seasonal = [74,72,78,69,74,78,74,71,76,73,77,74,79,63,60,67,58,63,66,63,61,65,62,67,62,68,52,51,57,47,52,56,53,49,54,52,55,51,57]

horizon = 4

# print(len(nonSeasonal), len(seasonal))

df = pd.DataFrame({'y': nonSeasonal})
df['unique_id'] = 1
df['ds'] = pd.date_range(start='2018-10-01', periods=len(df), freq='W')

fc = fc.Forecast(df['y'].values)

print(nonSeasonal[-horizon:])

es = fc.OptimalES().result[-horizon:]
ma = fc.MovingAverage(3).Print()
ma = fc.MovingAverage(4).Print()
ma = fc.MovingAverage(5).Print()
ma = fc.MovingAverage(6).Print()
ma = fc.MovingAverage(7).Print()
ma = fc.MovingAverage(13).Print()
lr = fc.LinearRegression().result[-horizon:]



# print(es)
# print(ma)
# print(lr)
