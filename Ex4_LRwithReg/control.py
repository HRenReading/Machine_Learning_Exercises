# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 18:57:33 2024

@author: 44754
"""

#import other .py files
from ReadFile import *
from plotData import *
from Validation import *
from Regularized_NLR import *
from utils import *


#read training & cross validation data, and count the numbers of samples and 
#features in the training set
x,y,xtest,ytest,xval,yval,m,n = read_data('ex5data1.mat')

###############################################################################
"""
"experiment with 1st-order single feature"
#update weights using linear regression
theta1 = linear_regression(x,y,m,n,Lambda)
#plot the data and linear regression with single feature
plotLinear(x,y,theta)
#compute the error of training and cv sets
error_train,error_val = LearningCurve(x,y,xval,yval,m,n)
#plot the learning curve of linear regression against the number of samples
plotLC(error_train,error_val,m)
"""
###############################################################################

"experiment with pth order features linear regression"
#updated weights, mean and std of the new features
theta2,mu,sigma = LR_HOPoly(x,y,m,p)
#plot the training data and polynomial function
plotPoly(x,y, mu, sigma, theta2, p, Lambda)



