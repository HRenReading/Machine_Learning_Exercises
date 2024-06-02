# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:44:46 2024

@author: 44754
"""

import matplotlib.pyplot as plt
import numpy as np


def plotData(ax, X, y):
    # Create New Figure
    # ====================== YOUR CODE HERE ======================
    # Find Indices of Positive and Negative Examples
    pos = y == 1
    neg = y == 0

    # Plot Examples
    ax.plot(X[pos, 0], X[pos, 1], 'k*', lw=2, ms=10)
    ax.plot(X[neg, 0], X[neg, 1], 'ko', mfc='y', ms=8, mec='k', mew=1)
     

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plotBoundary(X,y,clf,title):
    fig, ax = plt.subplots()
    # title for the plots
    # Set-up grid for plotting.
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    plotData(ax, X, y)    
    #ax.set_ylabel('y label here')
    #ax.set_xlabel('x label here')
    ax.set_xlim(np.min(X0),np.max(X0))
    ax.set_ylim(np.min(X1),np.max(X1))
    ax.set_title(title)
    #ax.legend()
    plt.show()