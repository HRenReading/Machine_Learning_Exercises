# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:31:34 2024

@author: 44754
"""

import numpy as np
from scipy.io import loadmat


def read_data(filename):
    "This is the function to read the data from .mat files"
    #the dictionary of the .mat file, including globals, version, header, 
    #samples X, and the labels, Y
    data = loadmat(filename)
    #abstract traning samples
    x,y = data['X'],data['y']
    #abstract the labels for the traning set
    xtest,ytest = data['Xtest'],data['ytest']
    xval,yval = data['Xval'],data['yval']
    #count the number of features and samples
    m,n = x.shape
    
    return x,y,xtest,ytest,xval,yval,m,n

