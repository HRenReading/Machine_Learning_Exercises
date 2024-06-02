# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:35:16 2024

@author: 44754
"""

#import packages and functions
import numpy as np
from scipy.io import loadmat
from scipy.optimize import fsolve,fmin
#import .py files
from plotData import *
from ReadFile import *

###############################################################################

def hypothesis(x,theta,n):
    """
    compute the output of our hypothesis
    
    Input -- x (m,n), the tranining data set. theta, parameters/weights.
    Output -- result of our hypothesis using Sigmoid function.
    """
    #make sure the parameters/weights are in the right form
    z = np.dot(x,theta)
    #compute the sigmoid function output
    g = 1./(1. + np.exp(-z))
    
    return g
    

def gradCost(thetab,x,y,m,n,Lambda):
    "compute the gradient of the cost function"
    #make a temporal theta with theta0 = 0
    T = np.copy(thetab).reshape((n+1,1))
    T[0,0] = 0
    #compute the our hypothesis output
    h = hypothesis(x,thetab,n).reshape(m,1)
    #covert dtype bool to int 0,1
    if y.dtype == bool:
        y = y.astype(int)
    #compute the cost function with regularization
    gradJ = np.dot(x.T,np.matrix(h-y))/m + np.dot(Lambda/m,T)
    
    return gradJ.A.flatten()

def multiclass_LogR(filename,Lambda,K):
    """
    Use the one-vs-all method with logistic regression to solve the multiclass
    classification problem.
    
    Input -- Lambda, the parameter for regularization. K, number of classes we
             want to separate the data.
    Output -- x, the training data with added x0 = 1. y, the labels of the data.
              theta_a, the updated parameters/weights that minimize the cost
              function. 
    """
    #read the data from .mat file
    x,y,m,n = read_data('ex3data1.mat')
    #create the matrix for all updated parameters/weights
    theta_all = np.zeros((K,n+1))
    #initialize parameters/weights
    thetab = np.zeros(n+1)
    #loop for 1VSall over K loops
    for k in range(K):
        theta_all[k,:] = fsolve(gradCost,thetab,args=(x,(y==k),m,n,Lambda))
        del k
    #find the indices of the maximum value in the columns
    p = np.argmax(hypothesis(x,theta_all.T,n), axis = 1)
    #print the accuracy in %
    print('Training Set Accuracy of Logistic regression: \
          {:.2f}%'.format(np.mean(p == y.reshape(m)) * 100))
          
    return x,y,m,n,theta_all




