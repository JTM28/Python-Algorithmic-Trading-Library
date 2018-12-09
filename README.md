# Python-Algorithmic-Trading-Library

A Library To Help Efficiently and Accurately Solve For Technical Indicators and Mathematical Functions Related To Quantitative Analysis of the Stock Market, Generate Trading Signals Through an Event Driven Backtest Engine, and Create Live Signals To Run Real Time With Python While Benefitting From C/C++ Computational speed (normally).


## Algorithmic Trading Indicators Goals:

1.  Attempting to solve all indicators in the most efficient way possible using python with the help of computing libraries
    such as numpy, scipy and many more. 
    
2. To create simple solutions for complex indicators and advanced mathematical functions that allow for a more 
   widespread use of advanced trading techniques


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
        
        
        
        
        
        
        
        
        

        
        
        

        
        
        
        
        
        
        
        
        
       
        
        
         
          
 


