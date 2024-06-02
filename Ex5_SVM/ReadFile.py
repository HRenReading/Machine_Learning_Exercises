# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import numpy as np
from scipy.io import loadmat


def read_Data(filename):
    "read the data from file and count the numbers of samples and features"
    if filename == 'ex6data3.mat':
        data = loadmat(filename)
        x,y = data['X'],data['y']
        y = y.astype(int)
        xval,yval = data['Xval'],data['yval']
        yval = yval.astype(int)
        m,n = x.shape
        
        return x,y,m,n,xval,yval
    else:
        data = loadmat(filename)
        x,y = data['X'],data['y']
        y = y.astype(int)
        m,n = x.shape
        
        return x,y,m,n
        
        
        
    