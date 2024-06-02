# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 11:44:24 2024

@author: 44754
"""

from ReadFile import *
from plotData import *

#read the features and labels from the .mat file and count the numbers of 
#samples&features
x,y,m,n = read_data('ex3data1.mat')
n_unit = 25
#number of classes
K = 10
#regularization parameter
Lambda = .1
#number of iteration used in minimizing the cost function
n_iter = None
"""
#random choice of index for plot
dice = np.random.choice(m, 100, replace=False)
show = x[dice,1:]
#plot the random samples
displayData(show)
"""