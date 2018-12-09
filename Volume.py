import math

import pandas as pd

import numpy as np

import scipy

from scipy import stats, fftpack

from scipy.fftpack import fft, ifft, fftfreq

pd.core.common.is_list_like = pd.api.types.is_list_like

import pandas_datareader as pdr

from pandas_datareader import *

import datetime as dt

import csv

import numba as nb

'''
-------------------------------------------------------------------------------------------------------------------------------------
Volume Based Signals

    *The Volume Signals Class Is Not Intended To Be An Exhaustive List of Volume Based Signals.
    
    **Volume Based Signals Tend To Work Best When Used As A Confirmation On TimeSeries Intervals

    Currently Supported Volume Signals:
    
        Chaikin Oscillator  |  Price Volume Trend
        
        Ease of Movement    |  Volume
        
        On-Balance Volume   |  Volume Oscillator
--------------------------------------------------------------------------------------------------------------------------------------
'''

class VolumeSignals():

    def __init__(self):

        self.selected_signal = None
        
        
    
    def CHAIKIN(high, low, close, volume, n_short, n_long):
        
        self.selected_signal = 'Chaikin Oscillator'

        df = pd.DataFrame({"Price" : close})

        df['MFM'] = np.round(((close - low) - (high - low)) / (high - low), 3)

        df['MFV'] = np.round(df['MFM'] * volume)

        df['ADL'] = np.round(df['MFM'].cumsum(), 3)

        df['FAST'] = np.round(df['ADL'].ewm(span= n_short).mean(), 3)

        df['SLOW'] = np.round(df['ADL'].ewm(span= n_long).mean(), 3)

        df['CHAIKIN'] = np.round(df['FAST'] - df['SLOW'], 4)

        return(df)


    def EOM(self, high, low, close, volume, n):
       
        self.selected_signal = 'Eease of Movement'

        eom_series = pd.DataFrame({"Price" : np.round(close, 2)})

        k = 100000000

        distance_moved = ((high + low) / 2) - ((high.shift() + low.shift()) / 2)

        box_ratio = (volume / k) / (high - low)

        eom_series['EOM'] = np.round((distance_moved / box_ratio), 3)

        eom_series['EOM_AVG'] = np.round(eom_series['EOM'].rolling(n).mean(), 3)

        return(eom_series)


    def OBV(ohlc_value, volume, n, n_smooth):

        df = pd.DataFrame({"Price" : ohlc_value, "Volume" : volume})

        df['OBV'] = np.where(df['Price'].diff() > 0, df['Volume'].rolling(n).sum(), -df['Volume'].rolling(n).sum())

        df['Line'] = df['OBV'].rolling(n_smooth).mean()

        return(df)


    def PVT(self, close, volume, n):

        self.selected_signal = 'Price Volume Trend'

        pvt_series = pd.DataFrame({"Price" : close})

        pvt_series['PVT'] = ((close.diff()) / (close.shift(1) * volume))

        pvt_series['PVT_AVG'] = pvt_series['PVT'].rolling(n).sum()

        return(pvt_series)


    def VOLUME(self, close, high, low, volume, n):

        self.selected_signal = 'Volume'

        oscillator = volume * (close - (high + low) / 2)

        cummulative_sum = oscillator.cumsum()

        rolling_sum = oscillator.rolling(n).sum()

        return(cummulative_sum, rolling_sum)


    def VO(self, volume, fast_n, slow_n):

        self.selected_signal = 'Volume Oscillator'

        fast_average = volume.rolling(fast_n).mean()

        slow_average = volume.rolling(slow_n).mean()

        diff = fast_average - slow_average

        oscillator = (diff / slow_average) * 100

        return(oscillator)
