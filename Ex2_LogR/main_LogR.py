# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:17:45 2024

@author: 44754
"""

from Logistic_regression import *
from Nonlinear_Logistic import *
from plotBoundary import *

###############################################################################

"first experiment for logistic regression with linear boundary"
x,y,theta,m,n = Logistic_regression('ex2data1.txt')
"plot the data and the linear boundary"
plotDecisionBoundary(plotData, theta, x, y)

###############################################################################
"second experiment for logistic regression with nonlinear boundary and the plot"
x2,y2,theta2,m2,n2 = Nonlinear_Log('ex2data2.txt',1)
"plot the data and the linear boundary"
plotDecisionBoundary(plotData, theta2, x2, y2)


