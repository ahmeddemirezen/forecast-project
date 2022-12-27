from statistics import mean
from unittest import result
import numpy as np
class Forecast:
    def __init__(self, _data):
        self.data = _data

    def MovingAverage(self, windowSize):
        result = []
        for i in range(len(self.data)):
            if i < windowSize:
                result.append(0)
            else:
                result.append(np.mean(self.data[i-windowSize:i]))
        
        mad = 0
        mse = 0
        mape = 0

        for i in range(len(self.data)):
            if(i < windowSize):
                continue
            mad += np.abs(self.data[i] - result[i]) 
            mse += np.square(self.data[i] - result[i]) 
            mape += np.abs(self.data[i] - result[i]) / self.data[i]
        
        mad = mad / (len(self.data) - windowSize)
        mse = mse / (len(self.data) - windowSize)
        mape = mape / (len(self.data) - windowSize)

        error = {"MAD": mad, "MSE": mse, "MAPE": mape}
        forecast = np.mean(self.data[-windowSize:])
        returnValue = Result(0,'Moving Average', result, error, {'step': windowSize}, forecast)
        return returnValue
    
    def OptimalMA(self):
        p_result, p_error = [] , {}
        p_step = 0
        p_forecast = 0
        for step in range(1, len(self.data)):
            temp = self.MovingAverage(step)
            result = temp.result
            error = temp.error
            forecast = temp.forecast
            step = temp.params['step']
            if(step == 1):
                p_result = result
                p_error = error
                p_step = step
                p_forecast = forecast
                continue
            resCounter = 0
            if(p_error["MAD"] > error["MAD"]):
                resCounter += 1
            if(p_error["MSE"] > error["MSE"]):
                resCounter += 1
            if(p_error["MAPE"] > error["MAPE"]):
                resCounter += 1
            if(resCounter >= 2):
                p_result = result
                p_error = error
                p_step = step
                p_forecast = forecast
        returnValue = Result(0, 'Optimal Moving Average', p_result, p_error, {'step': p_step}, p_forecast)
        return returnValue

    def ExponentialSmoothing(self, alpha):
        result = []
        for i in range(len(self.data)):
            if i == 0:
                result.append(self.data[i])
            else:
                result.append(alpha * self.data[i-1] + (1 - alpha) * result[i-1])
        
        mad = 0
        mse = 0
        mape = 0

        for i in range(len(self.data)):
            if(i == 0):
                continue
            mad += np.abs(self.data[i] - result[i]) 
            mse += np.square(self.data[i] - result[i]) 
            mape += np.abs(self.data[i] - result[i]) / self.data[i]
        
        mad = mad / (len(self.data) - 1)
        mse = mse / (len(self.data) - 1)
        mape = mape / (len(self.data) - 1)

        error = {"MAD": mad, "MSE": mse, "MAPE": mape}
        forecast = alpha * self.data[-1] + (1 - alpha) * result[-1]
        returnValue = Result(1 , 'Exponential Smoothing', result, error, {'alpha': alpha}, forecast)
        return returnValue

    def OptimalES(self):
        fResults = []
        for alpha in np.arange(0, 1, 0.01):
            fResults.append(self.ExponentialSmoothing(alpha))
        p_result, p_error = [] , {}
        p_step = 0
        p_forecast = 0
        for i in range(len(fResults)):
            result = fResults[i].result
            error = fResults[i].error
            forecast = fResults[i].forecast
            alpha = fResults[i].params['alpha']
            if(i == 0):
                p_result = result
                p_error = error
                p_step = alpha
                p_forecast = forecast
                continue
            if(p_error["MSE"] > error["MSE"]):
                p_result = result
                p_error = error
                p_step = alpha
                p_forecast = forecast
        returnValue = Result(1, 'Optimal Exponential Smoothing', p_result, p_error, {'alpha': p_step}, p_forecast)
        return returnValue

    # def OptimalES(self):
    #     fResults = []
    #     fResults.append(self.ExponentialSmoothing(0))
    #     fResults.append(self.ExponentialSmoothing(1))
    #     for iteration in range(10):
    #         fResults.sort(key=lambda x: x.params['alpha'])
    #         alpha = (fResults[1].params['alpha'] - fResults[0].params['alpha']) / 2
    #         fResults.append(self.ExponentialSmoothing(alpha))
    #         fResults.sort(key=lambda x: x.error['MSE'])
    #         print(fResults[0].error['MSE'], fResults[1].error['MSE'], fResults[2].error['MSE'])
    #         fResults.pop(2)
    #     returnValue = Result(1, 'Optimal Exponential Smoothing', fResults[0].result, fResults[0].error, {'alpha': fResults[0].params['alpha']}, fResults[0].forecast)
    #     return returnValue

    def LinearRegression(self):
        sum = 0
        expectedSum = 0
        mean = np.mean(self.data)
        for i in range(len(self.data)):
            sum += self.data[i]
            expectedSum += (i + 1) * self.data[i]
        n = len(self.data)
        SXY = (n * expectedSum) - (sum * n * (n + 1) / 2)
        SXX =  (n**2 * (n+1) * ((2*n) + 1) / 6) - (n**2 * (n+1) * (n+1) / 4)
        b = SXY / SXX
        a = mean - (b * (n+1) / 2)
        result = []
        for i in range(len(self.data)):
            result.append(a - b * (i+1))
        
        mad = 0
        mse = 0
        mape = 0

        for i in range(len(self.data)):
            if(i == 0):
                continue
            mad += np.abs(self.data[i] - result[i]) 
            mse += np.square(self.data[i] - result[i]) 
            mape += np.abs(self.data[i] - result[i]) / self.data[i]

        mad = mad / (len(self.data) - 1)
        mse = mse / (len(self.data) - 1)
        mape = mape / (len(self.data) - 1)

        error = {"MAD": mad, "MSE": mse, "MAPE": mape}
        forecast = a + b * (len(self.data) + 1)
        returnValue = Result(2, 'Linear Regression', result, error, {'a': a, 'b': b}, forecast)
        return returnValue

    def HoltsMethod(self, alpha, beta):
        result = []
        trend = []
        for i in range(len(self.data)):
            if i == 0:
                result.append(self.data[i])
                trend.append(0)
            else:
                result.append(alpha * self.data[i] + (1 - alpha) * (result[i-1] + trend[i-1]))
                trend.append(beta * (result[i] - result[i-1]) + (1 - beta) * trend[i-1])
        
        for i in range(len(self.data)):
            result[i] = result[i] + trend[i]
        
        print(trend)

        mad = 0
        mse = 0
        mape = 0

        for i in range(len(self.data)):
            if(i == 0):
                continue
            mad += np.abs(self.data[i] - result[i]) 
            mse += np.square(self.data[i] - result[i]) 
            mape += np.abs(self.data[i] - result[i]) / self.data[i]
        
        mad = mad / (len(self.data) - 1)
        mse = mse / (len(self.data) - 1)
        mape = mape / (len(self.data) - 1)

        error = {"MAD": mad, "MSE": mse, "MAPE": mape}
        forecast = result[-1] + trend[-1]
        returnValue = Result(3, 'Holts Method', result, error, {'alpha': alpha, 'beta': beta}, forecast)
        return returnValue

    def Solver(self):
        pass   

class Result:
    def __init__(self,_id , _name, _result, _error,_params, _forecast):
        self.id = _id
        self.name = _name
        self.result = _result
        self.error = _error
        self.params = _params
        self.forecast = _forecast

    def Print(self):
        print("Name: ", self.name)
        print("Result: ", self.result)
        print("Error: ", self.error)
        print("Paramters: ", self.params)
        print("Forecast: ", self.forecast)