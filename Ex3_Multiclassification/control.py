# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:26:54 2024

@author: 44754
"""
import time
from LogR_multiclass import *
from parameter import *
from Neural_Network import *

###############################################################################

#timeing the logistic regression
start_time1 = time.time()
#execution command for multiclassifier
x1,y1,m1,n1,theta_a1 = multiclass_LogR('ex3data1.mat',Lambda,K)
#print the time for running the Logistic Regression
print("--running Logistic regression %s seconds--" % (time.time() - start_time1))

###############################################################################
#timeing the neural net work with the same data
start_time2 = time.time()
#execution command for nueral network
x,y,m,n = OneHiddenLayer_NN('ex3data1.mat',K,n_unit,Lambda,n_iter)
print("--running Neural Network %s seconds--" % (time.time() - start_time2))




