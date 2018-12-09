# Python-Algorithmic-Trading-Library

A Library To Help Efficiently and Accurately Solve For Technical Indicators and Mathematical Functions Related To Quantitative Analysis of the Stock Market, Generate Trading Signals Through an Event Driven Backtest Engine, and Create Live Signals To Run Real Time With Python While Benefitting From C/C++ Computational speed (normally).

~Inspired by Daniel 



## Machine Learning Algorithms and Implemented Data Sources

While any datasource that works with pandas can be easily edited to work with the algorithms, there is current support
for Tiingo and Quandl SP-500 1 min Intervals*

If Using Quandl SP_500 Intervals The os.path will need to be changed internally if the quandl data is saved to a local
SSD or Hard Drive. 

*If Using Quandal Minute Intervals Stored locally, it is recommended to save it to the local drive to make for simpler access

         #Accessing Local Drive To Retrieve Quandl SP_500 1min Intervals Data
         import os
         
         os_denied = True
         
         while os_denied == True:
         
            try:      
                check_dir = os.chdir('C:\\Users\username\filename')
                new_cwd = os.getcwd()
                directory = os.listdir(new_cwd)
                
                if len(directory) > 0:
                     os_denied = False 
                     directory.sort()
                     
            except PermissionError:
                os_denied = True
             





Current Machine Learning Types:

Linear Regression Model
   Benefits - Easy To Implement, Simple to Adjust, Allows for continuous response variable
   Downside - Uses Best Fit Line, no exponential weighting to emphasize most current training data
   
Logistic Regression Model
   Benefits - Also easy to implement and adjust, allows for binary reinforcement learning ex.(np.where(x_cond == True, 1, -1))
   Downside - Difficult if using multiple logistic regression, difficulty finding indp. variable, limited number of outcomes



## Algorithmic Trading Indicators Structure

The signals are first broken down as either a technical indicator or formulas. Technical indicators are then
organized by their type ex.(Bollinger Bands == Volatility Signal) and formulas are grouped together by type of 
formulas used ex.(Inverse Fisher Transform == Bipolar Gaussian Normal Probability Distribution).

Current Technical Indicators:


#### Momentum
Chande Momentum Oscillator (CMO)
Momentum (MOM)
Percentage Price Oscillator (PPO)
Rate of Change (ROC)
Relative Strength Index (RSI)
Stochastic Fast/Slow (STOCH)

#### Trend
Double Exponential Moving Average (DEMA)
Exponential Moving Average (EMA)
Moving Average Channels (CHANNEL)
Simple Moving Average (SMA)
Triple Smoothed Exponetial Oscillator (TRIX)
True Strength Index (TSI)

#### Volatility
1. Average True Range (ATR)
2. Bollinger Bands (BOLLINGER)
3. Commodity Channel Index (CCI)
4. Kelter Channels (KELTER)
5. Standard Deviation (STDEV)

#### Volume
Accumulation Distrib. Line (ADL)
Chaikin Money Flow (CMF)
Ease of Movement (EOM)
On-Balance Volume (OBV)
Price Volume Trend (PVT)
Volume (VOLUME)
Volume Oscillator (VO)



## Signals Based On Mathematical Formulas

Types: 

Algebra
Interpolation
Matrix Math
Multivariable Calculus
Point Estimate Theory
Regression (Both Machine Learning and Auto-Correlation against Indp)


   

1.  Attempting to solve all indicators in the most efficient way possible using python with the help of computing libraries
    such as numpy, scipy and many more. 
    
2. To create simple solutions for complex indicators and advanced mathematical functions that allow for a more 
   widespread use of advanced trading techniques
   
## Importance of Normalization of Values When Writing A Signal For Multiple Securities
When trading a wide range of securities, a signal with set values hard coded into the filter may work well for one
ticker, but completely break when ran on a different security. This is due to a lack of normalization amongst the values.

In the following example, Both 30 minute SMA Values are exaclty $0.50 > SMA(60). However, there is a massive difference in the percentage difference of these two securities. 

The percent difference between the 30 and 60 minute SMA For SPY is less than a 0.2% difference while
the percent difference between the 30 and 60 minute SMA For AMD is over 2.7%
        
        spy_30min_sma = 265.5
        spy_60min_sma = 265.0        
        spy_sma_diff = (spy_30min_sma - spy_60min_sma)
        
        amd_30min_sma = 18.9
        amd_60min_sma = 18.4
        amd_sma_diff = (amd_30min_sma - amd_60min_sma)

To account for the wide range of share prices for different stocks, we can take the natural log of (value[i] / value[i-n])
which will return us a normalized value of the difference between the two values. This way all movements along a TimeSeries
axis will shows its real net movement and not be skewed by huge price differences amongst tickers allowing for signals 
with set values to be hard coded using the natural log value of whatever the difference of two values/indicators are.

        #Example Of Normalization Using Adjusted Returns
        
        import numpy as np
        
        df = pd.DataFrame({"Close":close})
       
        #Non-Normalized Adjusted Returns
        df['Adj'] = np.divide(np.subtract(df['Close'], df['Close'].shift(1)), df['Close'].shift(1))
        
        #Normalized Adjusted Returns
        df['Log-Adj'] = np.log(np.divide(df['Close'], df['Close'].shift(1)))
        
        

        
        
        
        
        
        
        

        


## How To Structure Indicators Within The Main Body Of Trading Systems Real Time Processing Engine

1. The indicators in this library show how to solve for signals using technical analysis, but in no way
   should these indicators be used to trade live with real money and risk at stake. These indicators are simple
   and basic, to actually solve for signals to trade live intraday, would require much more complex methods.
   
   **If interested in learning more about how to derive signals from the market:
     Khan Academy offers courses in areas such as linear algebra and multivariable calculus, and also in quantitative 
     statistics and how to interpret them. These are very important concepts to understand before moving forward. 
     For optimizing signals, learning about stochastic processes and the using of different forms of these processes 
     such as the Ornstein Uhlenbeck to backsolve to find the most optimized form of the signal
     
     
 ## How To Increase Computational Speed and Efficiency Through OpenSource Projects
 
 More than anything else, we will make use of the numpy and pandas import libraries everywhere throughout the library.
 It is these projects that will allow us to run computations in c/c++ using numpy and pandas without having to go through
 the much more frustrating process of writing c extensions.
 
 
        #Efficient Computations in only Numpy

        import numpy as np
        
        def ROC(df, n):
        
            close = np.array(df['Close'], dtype=np.float)
            
            roc = ((np.divide(np.subtract(close[n::1], close[0:len(close)-n:1]), close[0:len(close)-n:1]) * 100)
            
            return(roc)
        
        
        
        
        
        
        
        
        

        
        
        

        
        
        
        
        
        
        
        
        
       
        
        
         
          
 


