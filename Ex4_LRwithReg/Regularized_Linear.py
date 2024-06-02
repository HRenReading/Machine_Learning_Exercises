# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:33:18 2024

@author: 44754
"""
#import packages & functions in packages
import numpy as np
from scipy import optimize
#import other .py files
from ReadFile import *
from plotData import *
from parameter import *


def Cost_grad(theta,x,y,m,n,Lambda):
    """compute the cost function and its gradient
    
    Input -- theta, weights/parameters. x, training data features. y, trianing 
             data labels.
    Output -- the cost and value of cost function's gradient.
    """
    #make sure the weights are in a column vector not 1D array
    theta = theta.reshape((n+1,1))
    #make sure x & y are in the right matrix form
    x = x.reshape((m,n+1))
    y = y.reshape((m,1))
    #output of the linear hypothesis
    h = x @ theta
    #create a new vector of weights with theta0 = 1
    T = np.copy(theta)
    T[0,0] = 0
    #compute the cost with regularization
    J = 1./(2 * m) * (np.dot((h-y).T, (h-y)) + Lambda * np.dot(T.T, T))
    #compute the gradient of the cost function
    grad = 1./m * (np.dot(x.T, (h-y)) + Lambda * T)
    
    return J.reshape(1), grad.reshape(n+1)
    

def linear_regression(x,y,m,n,Lambda):
    """
    Linear regression for 1st order polynomial or higher order polynomial.
    
    Input -- filename, the file contains the data. Lambda, regularization
             parameter. key, the swith for either 1st order polynomial or 
             higher order polynomial.
    Output -- x, fearures of the training set. y, labels of the training set.
              theta_a, updated weights/parameters.
    """
    #read data from file and count the numbers of samples and features in the data
    xnew = np.concatenate((np.ones((m,1)),x),axis = 1)
    thetab = np.zeros(n+1)
    res = optimize.minimize(Cost_grad,thetab,args=(xnew,y,m,n,0.),\
                            jac=True, method='TNC', options = {'maxfun': 10*4})
    theta_a = res.x
    
    return theta_a




    
    


