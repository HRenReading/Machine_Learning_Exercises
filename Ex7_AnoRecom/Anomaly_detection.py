# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:33:32 2024

@author: 44754
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.linalg import inv,pinv
from scipy import stats

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
def read(filename):
    "read data from .mat files"
    #open the dictionary from the file
    data = loadmat(filename)
    #find the features of the training set and CV set, and the labels of the 
    #CV set
    x,xval,yval = data['X'],data['Xval'],data['yval']
    return x,xval,yval

def MGD(x):
    "fit the model to a multivariate Gaussian distribution"
    #count the numbers of samples and features
    m,n = x.shape
    #compute the meam
    mu = np.mean(x,axis=0)
    #compute the averaged covariance matrix
    diff = x-mu
    cov = diff.T @ diff / m
    #produce a MGD model
    MGDmodel = stats.multivariate_normal(mu, cov)
    return MGDmodel.pdf

###############################################################################
#read data from file
x,xval,yval = read('ex8data1.mat')
#train the MGD model
MGDmodel = MGD(x)
#scatter the training set
plt.figure()
ax, ay = np.mgrid[0:30:0.01, 0:30:0.01]
pos = np.dstack((ax, ay))
plt.contourf(ax,ay,MGDmodel(pos),cmap='Blues')
plt.scatter(x[:,0],x[:,1],marker='o',color='r',s=20)
plt.xlabel('Latency: ms')
plt.ylabel('Throughput: mb/s')
plt.show()

