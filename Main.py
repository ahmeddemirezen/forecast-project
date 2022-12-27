import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import Forecasts as fc

data = [35, 45, 50, 40, 45, 60, 70, 80, 90, 100, 110, 120]
data2 = [92,87,95,90,88,93]
data3 = [200,250,175,186,225,285,305,190]
data31 = [200,250,175,186,225]
data4 = [39,44,40,45,38,43,39]

forecast = fc.Forecast(data3)

forecast.HoltsMethod(0.1,0.1).Print()
    
        

# print(forecast.Solver())