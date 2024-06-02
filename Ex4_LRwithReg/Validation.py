# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 08:46:09 2024

@author: 44754
"""

import numpy as np
from Regularized_Linear import *

def LearningCurve(x,y,xval,yval,m,n):
    """
    Generates the train and cross validation set errors needed to plot a 
    learning curve returns the train and cross validation set errors for a
    learning curve. 
    
    Input -- x,y: features and labels of the training set. xval,yval: features
             and labels of the cross validation set. m, number of samples. 
             theta, updated weights with training set.
    Ouput -- error_train,error_val: train and cross validation set errors.
    """
    #create empty array to store trianing and cv error
    error_train = np.zeros(m)
    error_val = np.zeros(m)
    #add x0 = 1 to cv data
    xcv = np.concatenate((np.ones((yval.shape[0],1)),xval),axis=1)
    for i in range(m):
        thetaLC = linear_regression(x[:i+1,:],y[:i+1,:],i+1,n,0.)
        #make sure that weights are in a column vector
        thetaLC = thetaLC.reshape((n+1,1))
        #abstract the train & cv data of i+1 examples, and add x0 = 1
        xtrain = np.concatenate((np.ones((i+1,1)),x[:i+1,:]),axis=1)
        ytrain = y[:i+1,0]
        #calculate the error of training & cv data sets
        Jtrain = Cost_grad(thetaLC,xtrain,ytrain,i+1,n,0)[0]
        error_train[i] = Jtrain[0]
        Jcv = Cost_grad(thetaLC,xcv,yval,yval.shape[0],n,0)[0]
        error_val[i] = Jcv[0]
        del i
    return error_train,error_val
    
    
    
    