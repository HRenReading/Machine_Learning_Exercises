# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 18:06:39 2024

@author: 44754
"""
import numpy as np
import math

from scipy.linalg import inv,pinv
from scipy.optimize import fmin,fsolve

###############################################################################
def read(name_of_file):
    """
    read data from .txt files
    
    Input -- name_of_file, the name of the file.
    Output -- the data from the file, separated features and labels (x,y). 
              numbers of sample and feature size m,n.
    """
    #we need to eliminate the ',' or ';' in the text file for np.loadtxt to work
    data = np.loadtxt(name_of_file,delimiter=',')
    #count the number of features
    n = len(data[0,:-1])
    #count the number of samples
    m = len(data[:,0])
    #seperate the features and labels
    x = data[:,:n]
    #add an additional x0=1 to x
    #x = np.concatenate(np.ones(n), x)
    y = data[:,-1]
    return x.T,y.reshape((1,m)),m,n

def Sigmoid_h(x,theta):
    """
    compute the hypothesis as a Sigmoid function
    
    Input -- x, features of all samples. theta, parameters/weights of the features.
    Output -- h, result of our hypothesis
    """
    #compute the new vector 
    z = np.dot(theta.T,x)
    #Sigmoid equation
    h = 1/(1+np.exp(-z))
    """
    #set threshold for h so h will be either 1 or 0
    for i in range(len(h[0,:])):
        if h[0,i]>=0.5:
            h[0,i] = 1. 
        else:
            h[0,i] = 0
    """       
    return h



def Logistic_regression(filename):
    """
    Input -- the file contains the data
    Output -- x, features. y, labels. theta_a, updated parameters/weights. h, our
              hypothesis results.
    """
    #read the data from file and count the number of samples and features
    xold,y,m,n = read(filename)
    #add an additional feature x0 = 1
    x = np.concatenate((np.ones((1,m)), xold),axis = 0)
    #initialize parameters/weights
    thetab = np.zeros(n+1)
    #the cost function of the logistic regression
    def gradCost(theta_a):
        """
        Compute the gradient of the cost function
                          
        Input -- initial guess of the parameters/weights
        Output -- averaged cost
        """
        #convert theta to a matrix from a array
        theta_a = theta_a.reshape((n+1,1))
        #compute the hypothesis
        h = Sigmoid_h(x,theta_a)
        #compute the gradient of the cost function using vectorization
        gradJ = np.dot((h-y),x.T)/m
        
        return gradJ.reshape(n+1)
    
    theta_a = fsolve(gradCost,thetab,xtol =1e-8)
    
    return x.T,y.flatten(),theta_a,m,n



