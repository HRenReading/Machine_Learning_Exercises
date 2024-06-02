# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:41:40 2024

@author: 44754
"""
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score

from ReadFile import *
from plot import *

"""
###############################################################################

#read the data from file and count the numbers of samples and features in data
x,y,m,n = read_Data('ex6data1.mat')
#use the encoded svm in python
clf1 = svm.SVC(C = 100.,kernel='linear')
clf1.fit(x, y[:,0])
#plot the data and boundary
plotBoundary(x,y[:,0],clf1,title='Decision Boundary from SVM')

###############################################################################

#read the data from file and count the numbers of samples and features in data
x2,y2 = read_Data('ex6data2.mat')[:2]
#use the svm.SVC funtion in python
clf2 = svm.SVC(C = 100000, kernel = 'rbf')
clf2.fit(x2, y2[:,0])
#plot the data and boundary
plotBoundary(x2,y2[:,0],clf2,title='Decision Boundary from SVM')
"""
###############################################################################

#read the data from file and count the numbers of samples and features in data
x3,y3,m,n,xval,yval = read_Data('ex6data3.mat')
#use the svm.SVC funtion in python
clf3 = svm.SVC(C = 10**4., kernel = 'poly')
clf3.fit(x3, y3[:,0])
#plot the data and boundary
plotBoundary(x3,y3[:,0],clf3,title='Decision Boundary from SVM')
#make prediction using trained model 
predictions = clf3.predict(xval)
print('Trained model accuracy on CV data:',accuracy_score(yval,predictions)*100,\
      '%')












