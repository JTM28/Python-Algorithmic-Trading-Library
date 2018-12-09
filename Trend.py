import math

import pandas as pd

import numpy as np

import scipy

import numba as nb


class Trend():

    def __init__(self):

        self.selected_signal = None



    def AO(self, high, low, close, n_fast, n_slow, n_ema_smooth):

        self.selected_signal = 'Absolute Oscillator'

        df = pd.DataFrame({"Close" : close})

        df['TPP'] = np.divide(np.add(np.add(high, low), close), 3)

        df['FAST'] = df['TPP'].rolling(fast_n).mean()

        df['SLOW'] = df['TPP'].rolling(slow_n).mean()

        df['AO'] = np.subtract(df['FAST'], df['SLOW'])

        df['Smooth-AO'] = df['AO'].ewm(span=n_ema_smooth).mean()

        df = df.dropna().reset_index(drop=True)

        return(df)


    def APO(self, close, n_fast_ema, n_slow_ema):

        self.selected_signal = 'Absolute Price Oscillator W/ Exponential Averages'

        df = pd.DataFrame({"Close":close})

        df['Fast'] = df['Close'].ewm(span= n_fast_ema).mean()

        df['Slow'] = df['Close'].ewm(span= n_slow_ema).mean()

        df['APO'] = np.subtract(df['Fast'], df['Slow'])

        df['Log-APO'] = np.log(np.divide(df['Fast'], df['Slow']))

        df = df.dropna().reset_index(drop=True)

        return(df)


    def DEMA(self, close, n):

        self.selected_signal = 'Double Exponential Moving Average: EMA(EMA(Price))'

        decay = (2 / (n + 1))

        df = pd.DataFrame({"Price" : close})

        df['DEMA'] = df['Price'].ewm(span=n).mean().ewm(span=n).mean()

        df = df.dropna().reset_index(drop=True)

        return(df)

    
    def EMA(self, close, n):

        self.selected_signal = 'Exponential Moving Average: EMA(Price)'

        df = pd.DataFrame({"Close":close})

        decay = (2 / (n + 1))

        ema_series = pd.DataFrame({"Price" : np.round(close, 2)})

        ema_series['EMA'] = np.round(close.ewm(span= n).mean(), 2)

        return(ema_series)



    def MACD(self, close, fast_ema, slow_ema, n_signal_line):

        self.selected_signal = 'Moving Average Crossover Divergence'

        fast_decay = (2 / (fast_ema + 1))

        slow_decay = (2 / (slow_ema + 1))

        df = pd.DataFrame({"Price":close})

        df['Fast-EMA'] = close.ewm(span= fast_ema).mean()

        df['Slow-EMA'] = close.ewm(span = slow_ema).mean()

        df['MACD'] = np.subtract(df['Fast-EMA'], df['Slow-EMA'])

        df['Log-MACD'] = np.log(np.divide(df['Fast-EMA'], df['Slow-EMA']))

        df['Signal-Line'] = df['MACD'].ewm(span=n_signal_line).mean()

        df['Histogram'] = np.subtract(df['MACD'], df['Signal-Line'])

        df['Log-Histo'] = np.log(np.divide(df['MACD'], df['Signal-Line']))

        df = df.dropna().reset_index(drop=True)

        return(df)



    def SMA(self, value, n):

        self.selected_signal = 'Simple Moving Average'

        df = pd.DataFrame({"Value" : value})

        df['SMA'] = df['Value'].rolling(n).mean()

        df = df.dropna().reset_index(drop=True)

        return(df)



    def SMA_CHANNEL(self, value, n_sma, upper_percent, lower_percent):

        self.selected_signal = 'Moving Average Channels'

        df = pd.DataFrame({"Value" : value})

        df['SMA'] = df['Value'].rolling(n).mean()

        df['Upper'] = np.multiply(df['SMA'], upper_percent)

        df['Lower'] = np.multiply(df['SMA'], lower_percent)

        return(df)
