# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:45:14 2024

@author: 44754
"""

import matplotlib.pyplot as plt
import numpy as np
from ReadFile import *
from Regularized_NLR import *

###############################################################################           

#plot settings
plt.rc('font', size=18)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=20)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
plt.rc('legend', fontsize=18)    # legend fontsize
plt.rc('figure', titlesize=18)  # fontsize of the figure title

############################################################################### 

def plotLinear(x,y,theta):
    "scatter the data and plot the hypothesis"
    plt.figure()
    plt.plot(x, y, 'ro', ms=10, mec='k', mew=1.5)
    #if the function is 1st_order polynomial
    xnew = np.concatenate((np.ones((y.shape[0],1)),x),axis=1)
    plt.plot(x,np.dot(xnew,theta),color='black',lw=2,label='1st order polynomial')
    plt.xlabel('Change in water level: x')
    plt.ylabel('Water flowing out the dam: y')
    plt.legend(loc='upper left')
    plt.show()
    

def plotLC(error_train,error_val,m):
    "plot the learning curve"
    plt.figure()
    x = np.arange(1,m+1,1)
    plt.plot(x,error_train,color='blue',label='Trianing error')
    plt.plot(x,error_val,color='r',label='CV error')
    plt.xlabel('Number of samples: m')
    plt.ylabel('Error')
    plt.legend()
    plt.show()
    
def plotPoly(x,y, mu, sigma, theta, p, Lambda):
    """
    Plots a learned polynomial regression fit over an existing figure.
    Also works with linear regression.
    Plots the learned polynomial fit with power p and feature normalization (mu, sigma).

    Parameters
    ----------
    polyFeatures : func
        A function which generators polynomial features from a single feature.

    min_x : float
        The minimum value for the feature.

    max_x : float
        The maximum value for the feature.

    mu : float
        The mean feature value over the training dataset.

    sigma : float
        The feature standard deviation of the training dataset.

    theta : array_like
        The parameters for the trained polynomial linear regression.

    p : int
        The polynomial order.
    """
    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    t = np.arange(np.min(x) - 15, np.max(x) + 25, 0.05).reshape(-1, 1)

    # Map the X values
    X_poly = mapFeatures(t,p)[0]
    X_poly -= mu
    X_poly /= sigma

    # Add ones
    X_poly = np.concatenate([np.ones((t.shape[0], 1)), X_poly], axis=1)
    h = np.dot(X_poly, theta)
    # Plot
    plt.figure()
    plt.plot(x, y, 'ro', ms=10, mew=1.5, mec='k')
    plt.plot(t, h,color='black',label='Polynomial Prediction')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.title('Polynomial Regression Fit (lambda = %f)' % Lambda)
    plt.ylim([-20, 50])
    plt.legend()
    plt.show()

   
        
        
        
        
        
        