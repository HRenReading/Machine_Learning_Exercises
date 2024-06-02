# -*- coding: utf-8 -*-
"""
Created on Wed May  1 18:09:42 2024

@author: 44754
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.io import loadmat
from sklearn import cluster
from PIL import Image, ImageDraw

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
""" 
#read the data from .mat files
data = loadmat('ex7data2.mat')
x = data['X']
#choose the K-mean method and set the parameters for it
model = cluster.KMeans(n_clusters=3,init='random',n_init=10)
#train the model with data
clusters = model.fit_predict(x)
#separate the clusters for plot
label1 = (clusters == 0)
label2 = (clusters == 1)
label3 = (clusters == 2)
#plot the clustered data
plt.figure()
plt.scatter(x[label1,0],x[label1,1],color = 'r',s=50)
plt.scatter(x[label2,0],x[label2,1],color = 'green',s=50)
plt.scatter(x[label3,0],x[label3,1],color = 'black',s=50)
plt.title('K-Mean Clustering with 3 clusters')
plt.show()
"""
###############################################################################  
"this is the part to use K-mean to compress images"
#read the data from image
img = mpl.image.imread('bird_small.png')
# Divide by 255 so that all values are in the range 0 - 1
img /= 255
#convert 3d array to 2d
x = img.reshape((128*3,128))
#choose the K-mean method and set the parameters for it
model = cluster.KMeans(n_clusters=3,init='random',n_init=10)
#train the model with data
clusters = model.fit_predict(x)
    
"""
#pot the 
# Display the original image, rescale back by 255
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(img)
ax[0].set_title('Original')
ax[0].grid(False)
ax[1].imshow(img_com)
ax[1].set_title('Compressed')
ax[1].grid(False)
"""


