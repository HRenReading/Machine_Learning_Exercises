# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 11:45:30 2024

@author: 44754
"""

#import packages and functions
import numpy as np
from scipy.io import loadmat


def read_data(filename):
    "This is the function to read the data from .mat files"
    #the dictionary of the .mat file, including globals, version, header, 
    #samples X, and the labels, Y
    data = loadmat(filename)
    #abstract traning samples
    x = data['X']
    #abstract the labels for the traning set
    y = data['y']
    #conver 10 to 0
    y[y == 10] = 0
    #count the number of features and samples
    m,n = x.shape
    #add the additional fearure x0 = 1
    xnew = np.concatenate((np.ones((m,1)),x),axis = 1)
    
    return xnew,y,m,n

def read_weight(filename,thetaname1,thetaname2):
    "read the weights from .mat file"
    weights = loadmat(filename)
    theta1 = weights[thetaname1]
    theta2 = weights[thetaname2]
    theta2 = np.roll(theta2, 1, axis=0)
    return theta1,theta2
    
    

