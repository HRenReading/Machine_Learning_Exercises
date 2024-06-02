# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 13:45:01 2024

@author: 44754
"""

#import packages & functions in packages
import numpy as np
from scipy import optimize
from Regularized_Linear import *


def mapFeatures(x,p):
    """
    create features for the higher-order polynomial function.
    
    Input -- x, the original features (m,n). p, the highest order term we want.
    """
    #create the empty matrix for storing the new features
    xnew = np.zeros((x.shape[0],x.shape[1]*p))
    #count the number of original feature
    n = x.shape[1]
    #count the numbers of samples and new features
    m,N = xnew.shape
    #create the new features including the original feature
    for i in range(p):
        xnew[:,i*n:(i+1)*n] = np.power(x,i+1)
        del i
        
    return xnew,N

def Normalization(x):
    """
    Normalize (feature scaling) the new features.
    
    Input -- x,the new features data.
    Output -- xnorm, new features after feature-scaling.
    """
    m,n = x.shape
    #compute the mean of the new features
    mu = np.ones((m,1)) @ np.mean(x,axis=0).reshape((1,n))
    #compute the difference between X and its mean
    diff = x - mu
    #compute the std
    sigma = np.std(diff,axis=0,ddof=1)
    #compute the normalized features
    xnorm = diff/sigma 
    
    return xnorm,mu[0,:],sigma


def LR_HOPoly(x,y,m,p):
    """
    compute the linear regression with higher-order polynomial function.
    
    Input -- x, original features of training set. y, labels of training set.
             m, number of samples. p, the highest term in polynomial.
    Output -- theta_a, updated weights
    """
    X,N = mapFeatures(x,p)
    Xnorm,mu,sigma = Normalization(X)
    theta = np.zeros(N+1)
    Xnorm = np.concatenate((np.ones((m,1)),Xnorm),axis=1)
    res = optimize.minimize(Cost_grad,theta,args=(Xnorm, y, m, N, Lambda),\
                            jac=True, method='TNC')
    theta = res.x
    
    return theta,mu,sigma
    
    

