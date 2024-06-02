# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 11:29:26 2024

@author: 44754
"""

#import packages and functions
import numpy as np
from scipy import optimize
#import parameters
from parameter import *

###############################################################################

def sigmoid(z):
    """
    compute the sigmoid function.
    
    Input -- z, input of the sigmoid function.
    Output -- results from the sigmoid function
    """
    g = 1./(1. + np.exp(-z))
    
    return g

def gradSigmoid(z):
    "use the sigmoid function to compute its gradient"
    #compute the sigmoid function
    g = sigmoid(z)
    #compute the gradient which is dg/dz = g(1-g)
    grad = g*(1-g)
    return grad


def initial_theta(row,column):
    "initialize the theta"
    #parameter
    epsilon = np.sqrt(6)/np.sqrt(row+column)
    W = np.random.rand(row, column) * 2 * epsilon - epsilon
    
    return W
      

def roll_theta(theta1,theta2):
    "create a vector with theta1 (row after row) and theta2"
    theta_roll = np.concatenate((theta1.ravel(),theta2.ravel()))
    return theta_roll
    
        
def unroll_theta(theta_roll,m,n,K,n_unit):
    """
    unroll the vector that contains theta1 and theta2
    
    Input -- theta_roll, the vector contains theta1 and theta2. m&n, the numbers
             of samples and features of the training set. K, the number of 
             classes. n_unit, number of units in the hidden layer.
    Output -- the unrolled theta1&theta2
    """
    theta1 = np.reshape(theta_roll[:n_unit*(n+1)],(n_unit,n+1))
    theta2 = np.reshape(theta_roll[n_unit*(n+1):],(K,n_unit+1))
    
    return theta1, theta2


def Cost_grad(theta_roll,x,y,m,n,K,n_unit,Lambda):
    "forward model for neural network"
    theta1,theta2 = unroll_theta(theta_roll, m, n, K, n_unit)
    #reform y in to a matrix contains K sets of y in columns
    y_binary = np.eye(K)[y.reshape(m)]
    #output from input level
    a1 = np.copy(x)
    #input for hidden level
    z2 = x @ theta1.T
    #output from hidden layer
    a2 = np.concatenate((np.ones((m,1)),sigmoid(z2)),axis = 1)
    #input for output layer
    z3 = np.dot(a2,theta2.T)
    #output from the output layer
    a3 = sigmoid(z3)
    "cost function for the neural network"
    J = -1./m * np.sum(np.sum((y_binary*np.log(a3)) + \
                              ((1-y_binary)*np.log(1-a3))))  + \
        (Lambda/(2*m))*(np.sum(theta1[:,1:].T @ theta1[:,1:]) + \
                         np.sum(theta2[:,1:].T @ theta2[:,1:]))
    "backward model"
    #error from the final output against the labels
    Delta3 = a3 - y_binary 
    #evaluate the activation function gradient at z2
    gradS = np.concatenate((np.ones((m,1)),gradSigmoid(z2)),axis = 1)
    Delta2 = Delta3 @ theta2 * gradS
    #gradient cost function for theta1 and theta2
    gradTheta1 = Delta2[:,1:].T @ a1/m
    gradTheta1[:,1:] += (Lambda/m)*(theta1[:,1:])
    gradTheta2 = Delta3.T @ a2/m
    gradTheta2[:,1:] += (Lambda/m)*(theta2[:,1:])
    grad = roll_theta(gradTheta1,gradTheta2) 
                                                                              
    return J,grad


def OneHiddenLayer_NN(filename,K,n_unit,Lambda,n_iter):
    """
    use the neural net work to find the best fit parameters/weights for both 
    hidden layer and output layer. K, number of classes. n_unit, number of units
    in the hidden layer.
    Input -- the file name of the training data.
    Output -- the best fit parameters/weights and probability
    """
    #read the features and labels from the .mat file and count the numbers of 
    #samples&features
    x,y,m,n = read_data(filename)
    #initialize weights 
    theta1 = initial_theta(n_unit,n+1)
    theta2 = initial_theta(K,n_unit+1)
    #roll theta1 & theta2 for compute the gradient of cost function
    theta_roll = roll_theta(theta1,theta2)  
    #convert the rolled weights back to the original form
    theta1_a,theta2_a = unroll_theta(theta_roll,m,n,K,n_unit)
    #maximum iteration
    option= {'maxfun': n_iter}
    #minimize the cost function using optimize.minimize
    res = optimize.minimize(Cost_grad,theta_roll,args=(x,y,m,n,K,n_unit,Lambda),\
                            jac=True,method='TNC',options = option)
    theta_a_roll = res.x
    p = prediction(theta_a_roll,x,m,n,K,n_unit)
    print('Training Set Accuracy of Neural Network: \
          {:.2f}%'.format(np.mean(p == y.reshape(m)) * 100))

    return x,y,m,n

def prediction(theta_roll,x,m,n,K,n_unit):
    """
    predition with updated weights.
    
    theta_a, updated weights in a rolled vector. x
    """
    #unroll the parameter vector for neural network
    theta1, theta2 = unroll_theta(theta_roll,m,n,K,n_unit)
    # You need to return the following variables correctly
    p = np.zeros(m)
    h1 = sigmoid(np.dot(x, theta1.T))
    h2 = sigmoid(np.dot(np.concatenate([np.ones((m, 1)), h1], axis=1), theta2.T))
    p = np.argmax(h2, axis=1)
    
    return p   
    
###############################################################################

    
    
    