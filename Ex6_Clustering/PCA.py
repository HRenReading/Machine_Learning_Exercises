# -*- coding: utf-8 -*-
"""
Created on Thu May  2 09:26:11 2024

@author: 44754
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

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

def featureNormalize(X):
    """
    Normalizes the features in X returns a normalized version of X where the mean value of each
    feature is 0 and the standard deviation is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    """
    mu = np.mean(X, axis=0)
    X_norm = X - mu

    sigma = np.std(X_norm, axis=0, ddof=1)
    X_norm /= sigma
    return X_norm



def PCA(x):
    "Principal component analysis (PCA) for dimension reduction"
    #count the number of samples m and number of features n
    m,n = x.shape
    #compute the covariance matrix
    cov = np.dot(x.T,x)/m
    #perform Singular Value Decomposition
    u,s,v = np.linalg.svd(cov)
    #make 2D data to 1D
    z = x @ u[:,:1]
    return z.flatten()
    
    
    
#read the data from .mat files
data = loadmat('ex7data1.mat')
x = data['X']
#normalize data
x = featureNormalize(x)
m,n = x.shape
#compute the covariance matrix
cov = np.dot(x.T,x)/m
#perform Singular Value Decomposition
u,s,v = np.linalg.svd(cov)
#make 2D data to 1D
z = x @ u[:,:1]

"""
#  Visualize the example dataset
plt.plot(x[:, 0], x[:, 1], 'bo', ms=10, mec='k', mew=1)
plt.axis([0.5, 6.5, 2, 8])
plt.gca().set_aspect('equal')
plt.grid(True)
"""



