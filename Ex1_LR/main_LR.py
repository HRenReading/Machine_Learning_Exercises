# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:10:12 2024

@author: 44754
"""

from Linear_Regression import *

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

#experiments for single feature data     
x,y,theta,h = LiearRegression('ex1data1.txt')
"plot the data and our hypothesis"
plt.figure()
plt.scatter(x,y,marker='x',color = 'r',label = 'traninig data')
plt.plot(x[0,:],h,color = 'black',label='Linear regression')
plt.xlabel('Population in different cities: $10^4$')         
plt.ylabel('Profit of a food truck: \\$ $10^4$')   
plt.show()      
plt.legend()

############################################################################### 
#experiments for single feature data     
x2,y2,theta2,h2 = LiearRegression('ex1data2.txt')


