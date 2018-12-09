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

from tiingo import TiingoClient

from itertools import zip_longest

import numba as nb



# -- | Acceleration Bands | --

def ACCEL(high, low, close):

    accel_series = pd.DataFrame({"Price" : close})

    accel_series['Upper'] = np.round((high * (3 * ((high - low) / (high + low) / 2))) * 0.001, 3)

    accel_series['Mid'] = np.round((high + low) / 2, 3)

    accel_series['Lower'] = np.round((low * (3 * ((high - low) / (high + low) / 2))) * 0.001, 3)

    return(accel_series)



# -- | Accumulation Distribution Line | --

def ADL(df, n):

    money_flow_multiplier = ((close - low) - (high - low)) / (high - low)

    money_flow_volume = (money_flow_multiplier * volume)

    adl = money_flow_volume.rolling(n).sum()

    return(adl)



# -- | Autoregressive Linear Model | --

def ARCH(value, n):

    price = np.array(value)

    x = [i for i in range(0, len(value))]

    gradient = []
   
    line_intercept = []
   
    r = []

    rsqrd = []
   
    p = []
   
    error = []
   
    close = []

    for i in range(0, len(price) -n+1):

        slope, intercept, r_value, p_value, std_error = scipy.stats.linregress(x[i:i+n], value[i:i+n])

        gradient.append(np.round(slope, 3))

        line_intercept.append(np.round(intercept, 3))

        r.append(np.round(r_value, 3))

        sqrd = r_value ** 2

        rsqrd.append(np.round(sqrd, 3))

        p.append(np.round(p_value, 4))

        error.append(np.round(std_error, 3))

        close.append(np.round(value[i+n-1], 2))

    arch_series = pd.DataFrame({"Price" : close, "Gradient" : gradient, "Regress" : r, "Coeff" : rsqrd, "P-Stat" : p, "ERROR" : error})

    return(arch_series)

 


# -- | Absolute Price Oscillator | --

def APO(close, fast_ema, slow_ema):

    fast_decay = (2 / (fast_ema + 1))

    slow_decay = (2 / (slow_ema + 1)) 

    apo_series = pd.DataFrame({"Price" : np.round(close, 2)})

    apo_series['FAST'] = np.round(close.ewm(span= fast_ema).mean(), 2)

    apo_series['SLOW'] = np.round(close.ewm(span= slow_ema).mean(), 2)

    apo_series['APO'] = np.round(apo_series['FAST'] - apo_series['SLOW'], 3)

    apo_series['LONG'] = apo_series['APO'] > 0

    apo_series['SHORT'] = apo_series['APO'] < 0

    return(apo_series)



# -- | Adjusted Returns | --

def ADJ(close):

    adj_series = pd.DataFrame({"Price" : np.round(close, 2)})

    adj_series['ADJ'] = np.round(close.diff(), 3)

    adj_series['LOG'] = np.round(np.log(close) - np.log(close.shift(1)), 3)

    adj_series['PCT'] = np.round((((close - close.shift(1)) / close) * 100), 2)

    return(adj_series)




# -- | Average True Range | --

def ATR(close, high, low, n):

    atr_series = pd.DataFrame({"Price" : np.round(close, 2)})

    method_1 = high - low

    method_2 = high - close.shift()

    method_3 =  close.shift() - low

    methods = pd.DataFrame({"M1" : np.round(method_1, 3), "M2" : np.round(method_2, 3), "M3" : np.round(method_3, 3)})

    atr_series['TR'] = methods[["M1", "M2", "M3"]].max(axis= 1)

    atr_series['ATR'] = np.round(atr_series['TR'].rolling(n).mean(), 3)

    atr_series['N-ATR'] = np.round((atr_series['ATR'] / atr_series['Price']) * 100, 3)

    return(atr_series)





# -- | Absolute Oscillator | --

def AO(high, low, close, fast_n, slow_n, smooth_n):

    ao_series = pd.DataFrame({"Price" : np.round(close, 2)})

    ao_series['TPP'] = np.round((high + low + close) / 3, 2)

    ao_series['FAST'] = np.round(ao_series['TPP'].rolling(fast_n).mean(), 2)

    ao_series['SLOW'] = np.round(ao_series['TPP'].rolling(slow_n).mean(), 2)

    ao_series['AO'] = np.round(ao_series['FAST'] - ao_series['SLOW'], 3)

    ao_series['SMOOTH'] = np.round(ao_series['AO'].rolling(smooth_n).mean(), 3)

    return(ao_series)





# -- | Black Scholes Option Pricing Calls/Puts | --

def BLACK_SCHOLES(spot_price, strike_price, risk_free_rate, length_until_expiration, volatility, option_type):

    P = spot_price

    K = strike_price

    r = risk_free_rate

    T = length_until_expiration

    sigma = volatility

    d1 = (np.log(P / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    d2 = (np.log(P / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    call = (P * stats.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * stats.norm.cdf(d2, 0.0, 1,0))

    put = (K * np.exp(-r * T) * stats.norm.cdf(-d2, 0.0, 1.0) - S * stats.norm.cdf(-d1, 0.0, 1.0))

    if option_type == 0:

        return(call)

    if option_type == 1:

        return(put)




# -- | Bollinger Bands | --

def BOLLINGER(close, n, k):

    bollinger_series = pd.DataFrame({"Price" : np.round(close, 2)})

    bollinger_series['UPPER'] = np.round(close.rolling(n).mean() + (close.rolling(n).std() * k), 3)

    bollinger_series['MID'] = np.round(close.rolling(n).mean(), 3)

    bollinger_series['LOWER'] = np.round(close.rolling(n).mean() - (close.rolling(n).std() * k), 3)

    bollinger_series['WIDTH'] = np.round(((bollinger_series['UPPER'] - bollinger_series['LOWER']) / bollinger_series['UPPER']) * 100, 3)

    bollinger_series['LONG'] = bollinger_series['Price'] > bollinger_series['UPPER']

    bollinger_series['SHORT'] = bollinger_series['Price'] < bollinger_series['LOWER']

    return(bollinger_series)



# -- | Commodity Channel Index | --

def CCI(high, low, close, n):

    cci_series = pd.DataFrame({"Price" : np.round(close, 2)})

    cci_series['TPP'] = np.round((high + low + close) / 3, 2)

    constant = 0.015

    cci_series['SMA'] = cci_series['TPP'].rolling(n).mean()

    cci_series['DEV'] = np.round(abs(cci_series['TPP'] - cci_series['SMA']), 4)

    cci_series['AVG_DEV'] = cci_series['DEV'].rolling(n).mean()

    cci_series['CCI'] = np.round(((cci_series['TPP'] - cci_series['SMA']) / (0.015 * cci_series['AVG_DEV'])), 3)

    return(cci_series)





# -- | Chaikin Oscillator | --

def CHAIKIN(high, low, close, volume, n_short, n_long):

    chaikin_series = pd.DataFrame({"Price" : close})

    chaikin_series['MFM'] = np.round(((close - low) - (high - low)) / (high - low), 3)

    chaikin_series['MFV'] = np.round(chaikin_series['MFM'] * volume)

    chaikin_series['ADL'] = np.round(chaikin_series['MFM'].cumsum(), 3)

    chaikin_series['FAST'] = np.round(chaikin_series['ADL'].ewm(span= n_short).mean(), 3)

    chaikin_series['SLOW'] = np.round(chaikin_series['ADL'].ewm(span= n_long).mean(), 3)

    chaikin_series['CHAIKIN'] = np.round(chaikin_series['FAST'] - chaikin_series['SLOW'], 4)

    return(chaikin_series)



# -- | Channels Of Max/ Min Price | --

def CHANNEL(value, n, k):

    channel_series = pd.DataFrame({"Price": value})

    channel_series['Upper'] = close.rolling(n).mean() + (close.rolling(n).mean() * (n / 100))

    channel_series['Lower'] = close.rolling(n).mean() - (close.rolling(n).mean() * (n / 100))

    channel_series['LONG'] = channel_series['Price'] > channel_series['Upper']

    channel_series['SHORT'] = channel_series['Price'] < channel_series['Upper']

    return(channel_series)



# -- | Chandelier Exit | --

def CHANDELIER(atr, close, n, k):

    true_range = np.array(atr[0])

    price = np.array(close)

    long_exit = []

    short_exit = []

    for i in range(0, len(price) -21):

        amax = np.amax(price[i:i+n])

        amin = np.amin(price[i:i+22])

        tr_avg = np.mean(price[i:i+22])

        long_exit.append(amax - (3 * tr_avg))

        short_exit.append(amin + (3 * tr_avg))

    return(long_exit, short_exit)



# -- | Chaikin Money Flow | --

def CMF(high, low, close, volume, n):

    cmf_series = pd.DataFrame({"Price" : np.round(close, 2)})

    cmf_series['MFM'] = np.round(((close - low) - (high - close)) / (high - low), 3)

    cmf_series['MFV'] = np.round(cmf_series['MFM'] * volume, 2)

    cmf_series['CMF'] = np.round(cmf_series['MFV'].rolling(n).sum() / volume.rolling(n).sum(), 3)

    return(cmf_series)



# -- | Chande Momentum Oscillator | --

def CMO(close, n):

    price = np.array(close)

    pos = [];  neg = []

    for i in range(1, len(price)):

        if (price[i] - price[i-1]) > 0:

            pos.append(price[i] - price[i-1])

            neg.append(0)

        if (price[i] - price[i-1]) < 0:

            neg.append(abs(price[i] - price[i-1]))

            pos.append(0)

    pos_sum = [np.mean(pos[i:i+n]) for i in range(0, len(pos) -n+1)]

    neg_sum = [np.mean(neg[i:i+n]) for i in range(0, len(neg) -n+1)]

    momentum = np.divide(np.subtract(pos_sum, neg_sum), np.add(pos_sum, neg_sum)) * 100

    return(momentum)




# -- | Coppock Curve | --

def COPPOCK(close, short_n, long_n):

    short_roc = (((close - close.shift(short_n)) / close.shift(short_n)) * 100)

    long_roc  = (((close - close.shift(long_n)) / close.shift(long_n)) * 100)

    roc = pd.DataFrame({"S-ROC" : short_roc, "L-ROC" : long_roc})

    roc['ADD'] = short_roc + long_roc

    roc['WMA'] = ((roc['ADD'] * 7/28) + (roc['ADD'].shift(1) * 6/28) + (roc['ADD'].shift(2) * 5/28) + (roc['ADD'].shift(3) * 4/28) + (roc['ADD'].shift(4) * 3/28) +

                  (roc['ADD'].shift(5) * 2/28) + (roc['ADD'].shift(6) * 1/28))

                
    return(roc)




# -- | Crossover | --

def CROSSOVER(value_1, value_2):

    len_1 = len(value_1)

    len_2 = len(value_2)

    crossover = False

    if len_1 > len_2:

        diff = len_1 - len_2

        for i in range(1, len(value_1) - diff):

            if value_1[i] > value_2[i] and value_1[i-1] < value_2[i-1]:

                crossover = True

            elif value_1[i] < value_2[i] and value_1[i-1] > value_2[i-1]:

                crossover = True

            else:

                crossover = False

    elif len_1 < len_2:

        diff = len_2 - len_1

        for i in range(1, len(value_2) - diff):

            if value_1[i] > value_2[i] and value_1[i-1] < value_2[i-1]:

                crossover = True

            elif value_1[i] < value_2[i] and value_1[i-1] > value_2[i-1]:

                crossover = True

            else:

                crossover = False

    return(crossover)






# -- | Double Exponential Moving Average | --

def DEMA(close, n):

    decay = (2 / (n + 1))

    dema_series = pd.DataFrame({"Price" : close})

    dema_series['EMA'] = dema_series['Price'].ewm(span= n).mean()

    dema_series['DEMA'] = dema_series['EMA'].ewm(span= n).mean()

    return(dema_series)





# -- | Denmark Reversal Points | --

def DENMARK(open, high, low, close):

    O = np.array(open)

    H = np.array(high)

    L = np.array(low)

    C = np.array(close)

    pivot_point = []

    support = []

    resistance = []

    for o,h,l,c in zip(O,H,L,C):

        if o > c:

            x = np.add(np.add(h, np.multiply(2, l)), c)

        elif c > o:

            x = np.add(np.multiply(h, 2), np.add(l, c))

        else:

            if c == o:

                x = np.add(np.add(h, l), np.multiply(c, 2))


        pivot_point.append(np.divide(x, 4))

        support.append(np.subtract(np.divide(x, 2), h))

        resistance.append(np.subtract(np.divide(x, 2), l))

    return(pivot_point, support, resistance)



# -- | Derivatives | --

def DERIVATIVE(value, n):

    derivative_series = pd.DataFrame({"Y" : value})

    derivative_series["Y'"] = (value - value.shift(n)) / n

    derivative_series["Y''"] = (derivative_series["Y'"] - derivative_series["Y'"].shift(n)) / n

    return(derivative_series)



# -- | Donchian Channel | --

def DONCHIAN(high, low, close, n):

    donchian_series = pd.DataFrame({"Price" : np.round(close, 2)})

    tpp = (high + low) / 2

    donchian_series['SMA'] = np.round(tpp.rolling(n).mean(), 2)

    donchian_series['MAX-TR'] = np.round((high.rolling(n).max() - low.rolling(n).min()), 2)

    donchian_series['UPPER'] = np.round(donchian_series['SMA'] + (0.5 * donchian_series['MAX-TR']), 2)

    donchian_series['LOWER'] = np.round(donchian_series['SMA'] - (0.5 * donchian_series['MAX-TR']), 2)

    donchian_series['LONG'] = donchian_series['Price'] > donchian_series['UPPER']

    donchian_series['SHORT'] = donchian_series['Price'] < donchian_series['LOWER']

    return(donchian_series)




# -- | Ease of Movement | --

def EOM(high, low, close, volume, n):

    eom_series = pd.DataFrame({"Price" : np.round(close, 2)})

    k = 100000000

    distance_moved = ((high + low) / 2) - ((high.shift() + low.shift()) / 2)

    box_ratio = (volume / k) / (high - low)

    eom_series['EOM'] = np.round((distance_moved / box_ratio), 3)

    eom_series['EOM_AVG'] = np.round(eom_series['EOM'].rolling(n).mean(), 3)

    return(eom_series)




# -- | Exponential Moving Average | --

def EMA(close, n):

    decay = (2 / (n + 1))

    ema_series = pd.DataFrame({"Price" : np.round(close, 2)})

    ema_series['EMA'] = np.round(close.ewm(span= n).mean(), 2)

    return(ema_series)



#  -- | Fast Fourier Transform Extrapolation | --

def FFT_EXTRAPOLATION(x, n_predict):

    n = x.size

    n_harmonics = 10    
    
    t = np.arange(0, n)

    p = np.polyfit(t, x, 1)         

    x_detrended = x - p[0] * t  
    
    x_freqdom = scipy.fft(x_detrended)  

    f = scipy.fftpack.fftfreq(n) 

    frequencies = list(range(n)) 
    
    frequencies.sort(key = lambda i: np.absolute(f[i]))
 
    t = np.arange(0, n + n_predict)

    restored_sigma = np.zeros(t.size)

    for i in frequencies[:1 + n_harmonics * 2]:

        amplitude = np.absolute(x_freqdom[i]) / n  

        phase = np.angle(x_freqdom[i])          

        restored_sigma += amplitude * np.cos(2 * np.pi * f[i] * t + phase)

    return(restored_sigma + p[0] * t)





# -- | Fibonacci Pivot Points | --

def FIB_RETRACTMENT(high, low, close):

    pivot_point = ((high + low + close) / 3)

    k_1 = 0.382

    k_2 = 0.618

    k_3 = 1

    s_1 = pivot_point - ((high - low) * k_1)

    s_2 = pivot_point - ((high - low) * k_2)

    s_3 = pivot_point - ((high - low) * k_3)

    r_1 = pivot_point + ((high - low) * k_1)

    r_2 = pivot_point + ((high - low) * k_2)

    r_3 = pivot_point + ((high - low) * k_3)

    s = pd.DataFrame({"S1" : s_1, "S2" : s_2, "S3" : s_3})

    r = pd.DataFrame({"R1" : r_1, "R2" : r_2, "R3" : r_3})

    fibonacci_series = pd.DataFrame({"Close" : close})

    fibonacci_series['Signal 1'] = (fibonacci_series['Close'] - s["S1"].shift(3)) > (r["R1"].shift(3) - fibonacci_series['Close'])

    fibonacci_series['Signal 2'] = (fibonacci_series['Close'] - s["S2"].shift(5)) > (r["R2"].shift(5) - fibonacci_series['Close'])

    return(fibonacci_series)



# -- | Fibonnaci Support and Resistance | --

def FIBONACCI_SAR(close, n):

    k_1 = 0.382

    k_2 = 0.618

    k_3 = 1

    sar_series = pd.DataFrame({"Price" : close})

    sar_series['S1'] = close - (k_1 * (close.rolling(n).max() - close.rolling(n).min()))

    sar_series['S2'] = close - (k_2 * (close.rolling(n).max() - close.rolling(n).min()))

    sar_series['S3'] = close - (k_3 * (close.rolling(n).max() - close.rolling(n).min()))

    sar_series['R1'] = close - (k_1 * (close.rolling(n).max() - close.rolling(n).min()))

    sar_series['R2'] = close - (k_2 * (close.rolling(n).max() - close.rolling(n).min()))

    sar_series['R3'] = close - (k_3 * (close.rolling(n).max() - close.rolling(n).min()))

    return(sar_series)






# -- | Force Index | --

def FORCE(close, volume, n):

    force_series = pd.DataFrame({"Price" : np.round(close, 2)})

    force_series['Force'] = close.diff() * volume

    force_series['Force_AVG'] = force_series['Force'].rolling(n).mean()

    return(force_series)




#  -- | Inverse Fisher Transformation | --

def IFT(oscillator):

    ift = []

    filter = np.multiply(np.subtract(oscillator, 50), 0.1)

    for x in filter:

        ift.append((np.exp(np.multiply(2, x)) - 1) / (np.exp(np.multiply(2, x)) +1))
  
    return(ift)




# -- | Interpolation | --

def INTERPOLATION(close, n):

    gradient = (close - close.shift(n)) / n

    prediction = close + (gradient * n)

    p_interval = (((prediction - close) / n) / prediction) * 100

    real_interval = close.diff(1)

    signal_generate = pd.DataFrame()

    series = pd.DataFrame({"Gradient" : np.round(gradient, 3), "Prediction" : np.round(prediction, 2), "P Interval" : np.round(p_interval, 3), "Real Interval" : np.round(real_interval, 3)})

    series['Deviation'] = series['Real Interval'] - series['P Interval']

    series['Signal'] = series['Real Interval'] > series['P Interval']

    series['Close'] = np.round(close, 2)

    return(series)



# -- | Intraday Volatility | --

def IV(open, high, low, close, n):

    iv_series = pd.DataFrame({"Open" : open, "High" : high, "Low" : low, "Close" : close})

    iv_series['IV'] = np.round(high - low, 2)

    iv_series['N-IV'] = (iv_series['IV'] / close) * 100

    iv_series['AVG'] = iv_series['IV'].rolling(n).mean()

    iv_series['N-AVG'] = (iv_series['N-IV'].rolling(n).mean() / close.rolling(n).mean()) * 100

    return(iv_series)




# -- | Kelter Channels | --

def KELTER(high, low, close, n, k):

    kelter_series = pd.DataFrame({"Price" : close})

    kelter_series['EMA'] = close.ewm(span= n).mean()

    method_1 = high - low

    method_2 = high - close.shift()

    method_3 =  close.shift() - low

    methods = pd.DataFrame({"M1" : np.round(method_1, 3), "M2" : np.round(method_2, 3), "M3" : np.round(method_3, 3)})

    methods['TR'] = methods[["M1", "M2", "M3"]].max(axis= 1)

    kelter_series['ATR'] = np.round(methods['TR'].rolling(n).mean(), 3)

    kelter_series['UPPER'] = np.round((kelter_series['EMA'] + (k * kelter_series['ATR']), 2))

    kelter_series['LOWER'] = np.round((kelter_series['EMA'] - (k * kelter_series['ATR']), 2))

    kelter_series['LONG'] = kelter_series['Price'] > kelter_series['UPPER']

    kelter_series['SHORT'] = kelter_series['Price'] < kelter_series['LOWER']

    return(kelter_series)





def KST(close, k1, k2, k3, k4, n1, n2, n3, n4):

    kst_series = pd.DataFrame({"Price" : close})

    roc_1 = (((kst_series['Price'] - kst_series['Price'].shift(k1)) / kst_series['Price'].shift(k1)) * 100)

    roc_2 = (((kst_series['Price'] - kst_series['Price'].shift(k2)) / kst_series['Price'].shift(k2)) * 100)

    roc_3 = (((kst_series['Price'] - kst_series['Price'].shift(k3)) / kst_series['Price'].shift(k3)) * 100)

    roc_4 = (((kst_series['Price'] - kst_series['Price'].shift(k4)) / kst_series['Price'].shift(k4)) * 100)

    kst_series['KST'] = roc_1.rolling(n1).sum() + roc_2.rolling(n2).sum() * 2 + roc_3.rolling(n3).sum() * 3 + roc_4.rolling(n4).sum() * 4

    return(kst_series)






# -- | Linear Moving Average | --

def LMA(close):

    '''
        Needs Work


    '''

    lma_10 = []

    for i in range(9, len(close)):

        lma_10.append((close[i-9]*1/55)+(close[i-8]*2/55)+(close[i-7]*3/55)+(close[i-6]*4/55)+(close[i-5]*5/55)+(close[i-4]*6/55)+(close[i-3]*7/55)+(close[i-2]*8/55)+(close[i-1]*9/55)+(close[i]*10/55))

    diff = [(c - l) for c, l in zip(close[-50:], lma_10[-50:])]

    return(lma_10, diff)


# -- | Log Normalized Delta Movement Distribution | --

def LOG(close):

    '''  
        Needs Work

    '''

    log = np.array(np.log(close))

    log_diff = [(log[i] - log[i-1]) for i in range(1, len(close))]

    log_cbrt = np.array(np.cbrt(log))

    log_cbrt_sqrd = np.array(np.square(log_cbrt))

    return(log_cbrt_sqrd)




# -- | Moving Average Crossover Divergences | --

def MACD(close, fast_ema, slow_ema, n):

    fast_decay = (2 / (fast_ema + 1))

    slow_decay = (2 / (slow_ema + 1))

    macd_series = pd.DataFrame({"Price" : np.round(close, 2)})

    macd_series['FAST'] = np.round(close.ewm(span= fast_ema).mean(), 2)

    macd_series['SLOW'] = np.round(close.ewm(span = slow_ema).mean(), 2)

    macd_series['MACD'] = np.round(macd_series['FAST'] - macd_series['SLOW'], 3)

    macd_series['LINE'] = np.round(macd_series['MACD'].rolling(n).mean(), 3)

    macd_series['HISTO'] = np.round(macd_series['MACD'] - macd_series['LINE'], 3)

    macd_series['LONG'] =  ((macd_series['HISTO'] > macd_series['LINE']) & (macd_series['HISTO'] > 0))

    macd_series['SHORT'] = ((macd_series['HISTO'] < macd_series['LINE']) & (macd_series['HISTO'] < 0))

    return(macd_series)



# -- | Mass Index | --

def MASS(high, low, close, n, n_avg):

    mass_series = pd.DataFrame({"Price" : np.round(close, 2)})

    mass_series['TR'] = np.round(high - low, 2)

    mass_series['EMA1'] = np.round(mass_series['TR'].ewm(span= n).mean(), 3)

    mass_series['EMA2'] = np.round(mass_series['EMA1'].ewm(span= n).mean(), 3)

    mass_series['MASS'] = np.round(mass_series['EMA1'] / mass_series['EMA2'], 3)

    mass_series['AVG'] = np.round(mass_series['MASS'].rolling(n_avg).mean(), 3)

    return(mass_series)



# -- | MESA Sine Wave | --

def MESA(close):

    fast_limit = 0.5

    slow_limit = 0.05

    n = np.size(x, 1)

    smooth = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    smooth_1 = 0.0; smooth_2 = 0.0; smooth_3 = 0.0; smooth_4 = 0.0

    smooth_5 = 0.0; smooth_6 = 0.0; smooth_7 = 0.0

    detrend = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    detrend_1 = 0.0; detrend_2 = 0.0; detrend_3 = 0.0; detrend_4 = 0.0

    detrend_5 = 0.0; detrend_6 = 0.0; detrend_7 = 0.0

    Q1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    Q1_1 = 0.0; Q1_2 = 0.0; Q1_3 = 0.0; Q1_4 = 0.0

    Q1_5 = 0.0; Q1_6 = 0.0; Q1_7 = 0.0

    I1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    I1_1 = 0.0; I1_2 = 0.0; I1_3 = 0.0; I1_4 = 0.0

    I1_5 = 0.0; I1_6 = 0.0; I1_7 = 0.0

    I2 = [0.0, 0.0]

    I2_1 = 0.0; I2_2 = 0.0

    Q2 = [0.0, 0.0]

    Q2_1 = 0.0; Q2_2 = 0.0

    Re = [0.0, 0.0]

    Re_1 = 0.0; Re_2 = 0.0

    Im = [0.0, 0.0]

    Im_1 = 0.0; Im_2 = 0.0

    per = [0.0, 0.0]

    per_1 = 0.0; per_2 = 0.0

    sper = [0.0, 0.0]

    sper_1 = 0.0; sper_2 = 0.0

    phase = [0.0, 0.0]

    phase_1 = 0.0; phase_2 = 0.0

    jI = 0.0

    jQ = 0.0

    dphase = 0.0

    alpha = 0.0

    a = 0.0962

    b = 0.5769

    smooth_7 = (4 * x[i] + 3 * x[i-1] + 2 * x[i-2] + x[i-3]) * 0.1

    detrend_7 = (0.0962 * smooth_7 + 0.5769 * smooth_5 - 0.5769 * smooth_3 - 0.0962 * smooth_1) * (0.075 * per_1 + 0.54)

    Q1_7 = (0.0962 * detrend_7 + 0.5769 * detrend_5 - 0.5769 * detrend_3 - 0.0962 * detrend_1) * (0.075 * per_1 + 0.54)

    I1_7 = detrend_4

    jQ = (0.0962 * Q1_7 + 0.5769 * Q1_5 - 0.5769 * Q1_3 - 0.0962 * Q1_1) * (0.075 * per_1 + 0.54)

    jI = (0.0962 * I1_7 + 0.5769 * I1_5 - 0.5769 * I1_3 - 0.0962 * I1_1) * (0.075 * per_1 + 0.54)

    Q2_2 = Q1_7 + jI

    I2_2 = I1_7 - jQ

    Q2_2 = 0.2 * Q2_2 + 0.8 * Q2_1

    I2_2 = 0.2 * I2_2 + 0.8 * I2_1

    Re_2 = I2_2 * I2_1 + Q2_2 * Q2_1

    Im_2 = I2_2 * Q2_1 - Q2_2 * I2_1

    Re_2 = 0.2 * Re_2 + 0.8 * Re_1

    Im_2 = 0.2 * Im_2 + 0.8 * Im_1

    if (Im_2 != 0.0) and (Re_2 != 0.0):

        per_2 = 360.0/np.arctan(Im_2/Re_2)
   
    if per_2 > (1.5 * per_1):

        per_2 = 1.5 * per_1

    elif per_2 < (0.67 * per_1):

        per_2 = 0.67 * per_1

    if per_2 < 6.0:

        per_2 = 6.0

    elif per_2 > 50.0:

        per_2 = 50.0
    
    per_2 = 0.2 * per_2 + 0.8 * per_1

    sper_2 = 0.33 * per_2 + 0.67 * sper_1

    if I1_7 != 0.0:

        phase_2 = atan(Q1_7/I1_7)
    
    dphase = phase_1 - phase_2

    if dphase < 1.0:

        dphase = 1.0
    
    alpha = fastlimit / dphase

    if alpha < slowlimit:

        alpha = slowlimit

    for i in range(2, len(close)):

        output_1 = alpha * close[i] + (1.0 - alpha) * (alpha * close[i-1] + (1.0 - alpha))

        output_2 = 0.5 * alpha * output_1 + (1.0 - 0.5 * alpha) * (0.5 * alpha * (alpha * close[i-2] + (1.0 - alpha)) + (1.0 - 0.5 * alpha))

    smooth_1 = smooth_2;  smooth_2 = smooth_3;  smooth_3 = smooth_4;  smooth_4 = smooth_5;  smooth_5 = smooth_6;  smooth_6 = smooth_7
    
    detrend_1 = detrend_2;  detrend_2 = detrend_3;  detrend_3 = detrend_4;  detrend_4 = detrend_5;  detrend_5 = detrend_6;  detrend_6 = detrend_7

    Q1_1 = Q1_2;  Q1_2 = Q1_3;  Q1_3 = Q1_4;  Q1_4 = Q1_5;  Q1_5 = Q1_6;  Q1_6 = Q1_7
    
    I1_1 = I1_2;  I1_2 = I1_3;  I1_3 = I1_4;  I1_4 = I1_5;  I1_5 = I1_6;  I1_6 = I1_7
    
    I2_1 = I2_2
    
    Q2_1 = Q2_2
    
    Re_1 = Re_2
    
    Im_1 = Im_2
    
    per_1 = per_2
    
    sper_1 = sper_2
    
    phase_1 = phase_2

    return(output_1, output_2)






# -- | Minimum / Maximum Price Ranges | --

def MINMAX(df, n):

    min = close.rolling(n).min()

    max = close.rolling(n).max()

    min_max = pd.DataFrame({"Max" : max, "Min": min, "Price" : close})

    min_max['Signal'] = ((min_max['Max'] - min_max['Price']) / min_max['Max']) < ((min_max['Price'] - min_max['Min']) / min_max['Price'])

    return(min_max)




# -- | Momentum | --

def MOMENTUM(close, n):

    momentum = [(close[i] - close[i-n]) for i in range(n, len(close))]

    return(momentum)




# -- | On-Balance Volume | --

def OBV(close, volume, n, n_smooth):

    obv_series = pd.DataFrame({"Price" : close, "Volume" : volume})

    obv_series['OBV'] = np.where(obv_series['Price'].diff() > 0, obv_series['Volume'].rolling(n).sum(), -obv_series['Volume'].rolling(n).sum())

    obv_series['Line'] = obv_series['OBV'].rolling(n_smooth).mean()

    return(obv_series)




# -- | Pivot Points | --

def PIVOT_POINT(high, low, close):

    pp_series = pd.DataFrame({"Price" : np.round(close, 2)})

    pp_series['TPP'] = np.round((high + low + close) / 3, 2)

    pp_series['S1'] = np.round((2 * pp_series['TPP'] - high), 2)

    pp_series['S2'] = np.round((pp_series['TPP'] - high + low), 2)

    pp_series['S3'] = np.round(low - (2 * (high - pp_series['TPP'])), 2)

    pp_series['R1'] = np.round((2 * pp_series['TPP'] - low), 2)

    pp_series['R2'] = np.round(pp_series['TPP'] + (high - low), 2)

    pp_series['R3'] = np.round(high + (2 * (high - pp_series['TPP'])), 2)

    return(pp_series)





# -- | Price Momentum Oscillator | --

def PMO(close, n):

    decay = (2 / (n + 1))

    pmo_series = pd.DataFrame({"Price" : np.round(close, 2)})

    pmo_series['ROC'] = np.round((((close - close.shift(n)) / close.shift(n)) * 100), 3)

    pmo_series['PMO'] = np.round(pmo_series['ROC'].ewm(span= n).mean(), 3)

    pmo_series['LONG'] = ((pmo_series['PMO'] > pmo_series['PMO'].shift(1)) & (pmo_series['PMO'].shift(1) > pmo_series['PMO'].shift(2)))

    return(pmo_series)




# -- | Price Volume Trend | --

def PVT(close, volume, n):

    pvt_series = pd.DataFrame({"Price" : close})

    pvt_series['PVT'] = ((close.diff()) / (close.shift(1) * volume))

    pvt_series['PVT_AVG'] = pvt_series['PVT'].rolling(n).sum()

    return(pvt_series)



# -- | Rate of Change | --

def ROC(close, n):

    roc_series = pd.DataFrame({"Price" : np.round(close, 2)})

    roc_series['ROC'] = np.round((((close - close.shift(n)) / close) * 100), 3)

    roc_series['DELTA'] = np.round(roc_series['ROC'].diff(), 3)

    return(roc_series)



# -- | Relative Strength Index | --

def RSI(close, n, n_smooth, n_roc):

    rsi_series = pd.DataFrame({"Price" : np.round(close, 2)})

    try:

        diff = np.diff(close)

        seed = diff[:n+1]

        up = seed[seed>=0].sum() /n

        down = -seed[seed<0].sum() /n

        rs = (up / down)

        rsi = np.zeros_like(close)

        rsi[:n] = 100 - (100 / (1 + rs))

    except ZeroDivisionError:

        np.seterr(divide= 'ignore')

    for i in range(n, len(close)):

        delta = diff[i-1]

        if delta > 0:

            upval = delta

            downval = 0

        else:

            upval = 0

            downval = -delta

        try:

            up = (up * (n - 1) + upval) /n

            down = (down * (n - 1) + downval) /n

            rs = up/down

            rsi[i] = 100 - (100 / (1 + rs))

        except ZeroDivisionError:

            np.seterr(divide= 'ignore')

    rsi_series['RSI'] = np.round(rsi, 2)

    ift_filter = (rsi_series['RSI'] - 50) * 0.1

    rsi_series['IFT'] = np.round((np.exp(ift_filter * 2) - 1) / (np.exp(ift_filter * 2) +1), 3)

    rsi_series['SMOOTH'] = np.round(rsi_series['RSI'].rolling(n_smooth).mean(), 2)

    rsi_series['ROC'] = (((rsi_series['RSI'] - rsi_series['RSI'].shift(n_roc)) / rsi_series['RSI']) * 10)

    return(rsi_series)




# -- | Simple Moving Average | --

def SMA(df, n):

    sma = np.divide(close.rolling(n).sum(), n)

    return(sma)



# -- | SMA Channels | --

def SMA_CHANNEL(close, n, high_n, low_n):

    sma = [np.mean(close[i:i+n]) for i in range(0, len(close) -n+1)]

    upper_channel = np.multipy(sma, high_n)

    lower_channel = np.multiply(sma, low_n)

    return(upper_channel, lower_channel)




# -- | Standard Deviation | --

def STDEV(close, n):

    stdev_series = pd.DataFrame({"Price" : np.round(close, 3)})

    stdev_series['STD'] = np.round(close.rolling(n).std(), 3)

    stdev_series['N-STD'] = np.round(np.multiply(np.divide(close.rolling(n).std(), close), 100), 3)

    return(stdev_series)





# -- | Stochastic Oscillator | --

def STOCH(high, low, close, n, k_smooth, d_smooth):

    stoch_series = pd.DataFrame({"High" : np.round(high, 2), "Low" : np.round(low, 2), "Close" : np.round(close, 2)})

    min = low.rolling(n).min()

    max = high.rolling(n).max()

    stoch_series['Fast_K'] = np.round(100 * (close - min) / (max - min), 3)

    stoch_series['Slow_K'] = np.round(stoch_series['Fast_K'].rolling(k_smooth).mean(), 3)

    stoch_series['Slow_D'] = np.round(stoch_series['Slow_K'].rolling(d_smooth).mean(), 3)

    stoch_series['LONG'] = ((stoch_series['Slow_K'] < 10) & (stoch_series['Fast_K'] > stoch_series['Fast_K'].shift(1)))

    stoch_series['SHORT'] = ((stoch_series['Slow_K'] > 90) & (stoch_series['Fast_K'] < stoch_series['Fast_K'].shift(1)))

    return(stoch_series)




# -- | Stop-Loss | --

def STOP_LOSS(close, n_stop, n_flash):

    stop = n_stop / 100

    flash = n_flash / 100

    trailing_stop = []

    trailing_flash = []

    price = []

    for i in range(1, len(close)):

        if close[i] > close[i-1]:

            stoploss = close[i] * (1-stop)

            flashlimit = close[i] * (1-flash)

            trailing_stop.append(stoploss)

            trailing_flash.append(flashlimit)

            price.append(close[i])

        elif close[i] < close[i-1]:

            trailing_stop.append(stoploss)

            trailing_flash.append(flashlimit)

            price.append(close[i])

        else:

            if close[i] == close[i-1]:

                trailing_stop.append(stoploss)

                trailing_flash.append(flashlimit)

                price.append(close[i])

    stoploss_series = pd.DataFrame({"Price" : price, "Trail" : trailing_stop, "Flash" : trailing_flash})

    stoploss_series['Exit'] = stoploss_series['Price'] < stoploss_series['Trail']






# -- | Swing Index | --

def SWING(open, high, low, close):

    T = 100000

    method_1 = high - close.shift()

    method_2 = low - close.shift()

    method_3 = high - low

    df = pd.DataFrame({"Method 1": method_1, "Method 2": method_2, "Method 3": method_3})

    K = df[["Method 1", "Method 2"]].max(axis= 1)

    R = df[["Method 1", "Method 2", "Method 3"]].max(axis= 1)

    swing = 50 * ((close.shift() - close + (0.5 * (close.shift() - open.shift())) + (0.25 * (close - open))) / R) * (K / T)

    return(swing)




# -- | Sine Weighted Moving Averages | --

def SWMA_7(close):

    swma = []

    for i in range(6, len(close)):

        swma.append(
            
             (
             (close[i]* np.sin(7/(7+1))*np.pi)+(close[i-1]*np.sin(6/(7+1))*np.pi)+(close[i-2]*np.sin(5/(7+1))*np.pi)+

             (close[i-3]*np.sin(4/(7+1))*np.pi)+(close[i-4]*np.sin(3/(7+1))*np.pi)+(close[i-5]*np.sin(2/(7+1))*np.pi)+

             (close[i-6]*np.sin(1/(7+1))*np.pi))
             /  
            
             (               
             (np.sin(7/(7+1))*np.pi)+(np.sin(6/(7+1))*np.pi)+(np.sin(5/(7+1))*np.pi)+

             (np.sin(4/(7+1))*np.pi)+(np.sin(3/(7+1))*np.pi)+(np.sin(2/(7+1))*np.pi)+

             (np.sin(1/(7+1))*np.pi)))

    return(swma)


def SWMA_12(close):

    swma = []

    for i in range(11, len(close)):

        swma.append(
            
             (
             (close[i]* np.sin(12/(12+1))*np.pi)+(close[i-1]*np.sin(11/(12+1))*np.pi)+(close[i-2]*np.sin(10/(12+1))*np.pi)+

             (close[i-3]*np.sin(9/(12+1))*np.pi)+(close[i-4]*np.sin(8/(12+1))*np.pi)+(close[i-5]*np.sin(7/(12+1))*np.pi)+

             (close[i-6]*np.sin(6/(12+1))*np.pi)+(close[i-7]*np.sin(5/(12+1))*np.pi)+(close[i-8]*np.sin(4/(12+1))*np.pi)+

             (close[i-9]*np.sin(3/(12+1))*np.pi)+(close[i-10]*np.sin(2/(12+1))*np.pi)+(close[i-11]*np.sin(1/(12+1))*np.pi))
                                            
             /  
            
             (               
             (np.sin(12/(12+1))*np.pi)+(np.sin(11/(12+1))*np.pi)+(np.sin(10/(12+1))*np.pi)+

             (np.sin(9/(12+1))*np.pi)+(np.sin(8/(12+1))*np.pi)+(np.sin(7/(12+1))*np.pi)+

             (np.sin(6/(12+1))*np.pi)+(np.sin(5/(12+1))*np.pi)+(np.sin(4/(12+1))*np.pi)+

             (np.sin(3/(12+1))*np.pi)+(np.sin(2/(12+1))*np.pi)+(np.sin(1/(12+1))*np.pi))                      
             )
                            
    return(swma)



# -- | Triple Smoothed Exponential Moving Average | --

def TRIX(close, k_1, k_2, k_3):

    trix_series = pd.DataFrame({"Price" : close})

    trix_series['EMA-1'] = trix_series['Price'].ewm(span= k_1).mean()

    trix_series['EMA-2'] = trix_series['EMA-1'].ewm(span= k_2).mean()

    trix_series['EMA-3'] = trix_series['EMA-2'].ewm(span= k_3).mean()

    trix_series['TRIX'] = (((trix_series['EMA-3'] - trix_series['EMA-3'].shift(1)) / trix_series['EMA-3']) * 10000)

    return(trix_series)



# -- | True Strength Indicator | --

def TSI(close, k_1, k_2):

    tsi_series = pd.DataFrame({"Price" : close})

    diff = close.diff()

    abs_diff = abs(close.diff())

    ema_1 = diff.ewm(span= k_1).mean() 

    abs_ema_1 = abs_diff.ewm(span= k_1).mean()

    ema_2 = ema_1.ewm(span= k_2).mean()

    abs_ema_2 = abs_ema_1.ewm(span= k_2).mean()

    tsi_series['TSI'] = (ema_2 / abs_ema_2) * 100

    return(tsi_series)


# -- | Ultimate Oscillator | --

def UO(df, short_n, med_n, long_n, smooth_n):

    buy_pressure = close - np.minimum(low, close.shift())

    buy_pressure = pd.Series(buy_pressure)

    true_range = np.maximum(high, close.shift()) - (np.minimum(low, close.shift()))

    true_range = pd.Series(true_range)

    short_average = buy_pressure.rolling(short_n).sum() / true_range.rolling(short_n).sum()

    med_average = buy_pressure.rolling(med_n).sum() / true_range.rolling(med_n).sum()

    long_average = buy_pressure.rolling(long_n).sum() / true_range.rolling(long_n).sum()

    oscillator = ((short_average * 4) + (med_average * 2) + (long_average * 2)) / 7

    smoothed_oscillator = oscillator.rolling(smooth_n).mean()

    return(oscillator, smoothed_oscillator)



# -- | Velocity Movement | --

def VELOCITY(close):

    direction = []

    roc = []

    for i in range(7, len(close)):

        roc.append(((abs(close[i] - close[i-7]) / close[i-7]) * 100))

        if close[i] > close[i-3] > close[i-7]:

            direction.append(2)

        else:

            if close[i] > close[i-7]:

                direction.append(1)

            else:

                if close[i] < close[i-3] < close[i-7]:

                    direction.append(-2)

                else:

                    if close[i] < close[i-7]:

                        direction.append(-1)

    velocity = np.multiply(direction[-50:], roc[-50:])

    return(velocity)




# -- | Volume Accumulation | --

def VOLUME(close, high, low, volume, n):

    oscillator = volume * (close - (high + low) / 2)

    cummulative_sum = oscillator.cumsum()

    rolling_sum = oscillator.rolling(n).sum()

    return(cummulative_sum, rolling_sum)





# -- | Volume Oscillator | --

def VO(volume, fast_n, slow_n):

    fast_average = volume.rolling(fast_n).mean()

    slow_average = volume.rolling(slow_n).mean()

    diff = fast_average - slow_average

    oscillator = (diff / slow_average) * 100

    return(oscillator)



# -- | Vortex Indicator | --

def VORTEX(high, low, close, n, smooth_n):

    with warnings.catch_warnings():

        warnings.simplefilter('ignore')

        vortex_series = pd.DataFrame({"High" : high, "Low" : low, "Price" : close})

        true_range = (np.maximum(high, close.shift(1))) - (np.minimum(low, close.shift(1)))

        v_range = abs(high - low.shift(1)) - abs(low - high.shift(1))

        vortex_series['Vortex'] = v_range.rolling(n).sum() / true_range.rolling(n).sum()

        vortex_series['VI-AVG'] = vortex_series['Vortex'].ewm(span= smooth_n).mean()

        return(vortex_series)






'''
    -------------------------------------------------------------------------

        Indicators and Formula Classes

            1. Volume Based Indicators

            2. True Momentum Indicators

            3. Overbought / Oversold Momentum Indicators

            4. Trend Based Indicators

            5. Trend Based Pattern Recognition

    -------------------------------------------------------------------------
'''





class Normalization():

    def __init__(self):

        pass

    def Difference(value, n):

        v = np.array(value, dtype= np.float)

        n_diff = np.log(np.divide(v[n:], v[0:len(v)-n]))

        return(n_diff)


    def Exponential(value, n):

        v = np.array(value, dtype= np.float)

        n_exp = np.log(np.divide(v[n:], v[0:len(v)-n])) * np.exp(n)

        return(n_exp)


    def Derivative(value, n):

        v = np.array(value, dtype=np.float)

        n_derivative = np.log(np.divide(v[n:], v[0:len(v)-n])) * 100

        return(n_derivative)















import math

import pandas as pd

import numpy as np

import scipy

from scipy import stats 

from scipy.stats import norm

from scipy import *

from scipy.stats import linregress

from math import floor

import scipy.fftpack

from scipy.fftpack import fft, ifft, fftfreq

import matplotlib.pyplot as plt

import sklearn

from sklearn import metrics

from scipy import special



# -- | BASIC DERIVATIVE | --

def DERIVATIVE(x, n):

    y = x ** n

    y1 = n*(x**n-1)




# ----- || Derivative Functions || -----

def ARCSIN(x):

    y = np.arcsin(x)

    y1 = 1 / (np.sqrt(1 - x.x))

    return(y, y1)



def ARCTAN(x):

    y = np.arctan(x)

    y1 = 1 / (1 + (x ** 2))

    return(y, y1)


def COS(x):

    y = np.cos(x)

    y1 = -np.sin(x)

    return(y, y1)


def COT(x):

    y = scipy.special.cotdg(x)

    y1 = -(1 / np.sin(x**2))

    return(y, y1)

def EXP(x):

    y = np.exp(x)

    y1 = np.exp(x)

    return(y, y1)


def LN(x):

    y = np.log(x)

    y1 = 1 / x

    return(y, y1)


def SIN(x):

    y = np.sin(x)

    y1 = np.cos(x)

    return(y, y1)


def TAN(x):

    y = np.tan(x)

    y1 = 1 / np.cos(x**2)

    return(y, y1)


def SINH(x):

    y = np.sinh(x)

    y1 = np.cosh(x)

    return(y, y1)


def COSH(x):

    y = np.cosh(x)

    y1 = np.sinh(x)

    return(y, y1)



def SECH(x):

    y = (1 / np.cosh(x))

    y = -((1/np.cosh(x))*np.tanh(x))

    return(y, y1)

    
def TANH(x):

    y = np.tanh(x)

    y1 = (np.sinh(x) / np.cosh(x))

    return(y, y1)


            
            
            
            