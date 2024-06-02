# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:08:45 2024

@author: 44754
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin
from scipy.linalg import pinv

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

def hypothesis(x,theta):
    """
    Hypothesis function for linear regression.
    
    Input -- x, matrix of n(number of features) * m(size of the data set).
             theta , parameters (size of n+1) of the hypothesis.
    Output -- y, the output of the hypothesis.
    """
    #linear regression hypothesis
    y = np.dot(theta.T,x)
    
    return y


def LiearRegression(filename):
    """
    perform linear regression with gradient descent
    
    Input -- x, matrix of n(number of features) * m(size of the data set).
             Lambda, regularization parameter. a, parameter to adjust initial
             parameters/weights.
    Output -- parameters that minimize the cost function
    """
    #read data from a file
    x_old,y,m,n = read(filename)
    #initialize parameters/weights
    thetab = np.ones((n+1,1))
    #if n == int:
    #reformulate the features by adding an extra feature vector x0=1
    x = np.concatenate((np.ones((1,m)), x_old),axis = 0)
    def cost_function(theta_a):
        #loop for summarizing the costfunction of all samples in the training set
        J = np.dot((np.dot(theta_a.T,x) - y),(np.dot(theta_a.T,x) - y).T)
        #sum over all the samples and take the average
        aveJ = np.sum(J)/(2*m)
             
        return aveJ
    #minimize the cost function using gradient descent
    theta_a = fmin(cost_function,thetab,xtol=1e-6,maxiter = 10**3)
    h_theta = hypothesis(np.concatenate((np.ones((1,m)), x_old),axis = 0),theta_a)
    
    return x_old,y,theta_a,h_theta



###############################################################################

   
             