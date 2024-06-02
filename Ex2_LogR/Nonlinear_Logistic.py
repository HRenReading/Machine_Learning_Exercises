# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:06:35 2024

@author: 44754
"""

import numpy as np
import math
import pandas as pd
from scipy.linalg import inv,pinv
from scipy.optimize import fmin,fsolve

###############################################################################
def read(name_of_file):
    """
    read data from .txt files
    
    Input -- name_of_file, the name of the file.
    Output -- the data from the file, separated features and labels (x,y). 
              numbers of sample and feature size m,n.
    """
    #we need to eliminate the ',' or ';' in the text file for np.loadtxt to work
    data = np.loadtxt(name_of_file,delimiter=',')
    #count the number of features
    n = len(data[0,:-1])
    #count the number of samples
    m = len(data[:,0])
    #seperate the features and labels
    x = data[:,:n]
    #add an additional x0=1 to x
    #x = np.concatenate(np.ones(n), x)
    y = data[:,-1]
    return x.T,y.reshape((1,m)),m,n

def Sigmoid_h(x,theta):
    """
    compute the hypothesis as a Sigmoid function
    
    Input -- x, features of all samples. theta, parameters/weights of the features.
    Output -- h, result of our hypothesis
    """
    #compute the new vector 
    z = np.dot(theta.T,x)
    #Sigmoid equation
    h = 1/(1+np.exp(-z))
    """
    #set threshold for h so h will be either 1 or 0
    for i in range(len(h[0,:])):
        if h[0,i]>=0.5:
            h[0,i] = 1. 
        else:
            h[0,i] = 0
    """       
    return h

def mapFeature(X1, X2, degree = 6):
    """
    Maps the two input features to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Parameters
    ----------
    X1 : array_like
        A vector of shape (m, 1), containing one feature for all examples.

    X2 : array_like
        A vector of shape (m, 1), containing a second feature for all examples.
        Inputs X1, X2 must be the same size.

    degree: int, optional
        The polynomial degree.

    Returns
    -------
    : array_like
        A matrix of of m rows, and columns depend on the degree of polynomial.
    """
    if X1.ndim > 0:
        out = [np.ones(X1.shape[0])]
    else:
        out = [np.ones(1)]

    for i in range(1, degree + 1):
        for j in range(i + 1):
            out.append((X1 ** (i - j)) * (X2 ** j))

    if X1.ndim > 0:
        return np.stack(out, axis=1)
    else:
        return np.asarray(out, dtype="object")


def Sigmoid_h(x,theta):
    """
    compute the hypothesis as a Sigmoid function
    
    Input -- x, features of all samples. theta, parameters/weights of the features.
    Output -- h, result of our hypothesis
    """
    #compute the new vector 
    z = np.dot(theta.T,x)
    #Sigmoid equation
    h = 1/(1+np.exp(-z))
    
    return h

def Nonlinear_Log(filename,Lambda):
    """
    Input -- the file contains the data
    Output -- x, features. y, labels. theta_a, updated parameters/weights. h, our
              hypothesis results.
    """
    #read the data from file and count the number of samples and features
    xold,y,m,n = read(filename)
    #create new features based on the original features
    x = mapFeature(xold[0,:], xold[1,:]).T
    #count the new feature size
    n = len(x[:,0])-1
    #initialize parameters/weights
    thetab = np.zeros(n+1)
    #the cost function of the logistic regression
    def gradCost(theta_a):
        """
        Compute the gradient of the cost function
                          
        Input -- initial guess of the parameters/weights
        Output -- averaged cost
        """
        #convert theta to a matrix from a array
        theta_a = theta_a.reshape((n+1,1))
        T = np.copy(theta_a)
        T[0,0] = 0
        #compute the hypothesis
        h = Sigmoid_h(x,theta_a)
        theta_a = theta_a.reshape(n+1)
        #compute the gradient of the cost function using vectorization
        gradJ = np.reshape(np.dot((h-y),x.T)/m,(n+1)) + Lambda/m*T.reshape(n+1)
        
        return gradJ.reshape(n+1)
    
    theta_a = fsolve(gradCost,thetab,xtol =1e-8)
    
    return x.T,y.flatten(),theta_a,m,n



