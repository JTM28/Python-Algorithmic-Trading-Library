import math

import numpy as np

import scipy

from scipy import stats 

from scipy.stats import norm

from scipy.stats import linregress

from math import floor

import scipy.fftpack

from scipy.fftpack import fft, ifft, fftfreq

import matplotlib.pyplot as plt



'''
----------------------------------------------------------------------------------------------------------------------------------------
Class For Finding The First Derivative Across A Time Series

Current Derivatives: 

--First Derivatives
Arccos  Cosine  SECH  SINH  Sine  COT  Tan
Arcsin  Arctan  TANH  COSH  Exp   LN


How To Use: 

Within the Derivative Class, input x and n into whatever derivative function you wish to solve.

Requirements: 
    
      1. x Must be either an int or float iterable array or python list
  




Requirements:
  
  * x = values
  * n = t**(i-n) with n being the offset 
  
  
  Note: Derivative Functions only Solve for y'
        **To Solve For y'' run the same function used to find y' on the values from y' to solve y''
        
        ex. y'= (y2 - y1) / n
        
        ex. y''= (y'[i] - y'[i-n]) / n
        
        
   **If Using Derivative Functions on non normalized values make sure to use the normilzation class
  -------------------------------------------------------------------------------------------------------------------------------------
'''      
      

class Derivatives():

    def __init__(self):

        pass


    def DERIVATIVE(x, n):

        y = x ** n

        y1 = n*(x**n-1)


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

