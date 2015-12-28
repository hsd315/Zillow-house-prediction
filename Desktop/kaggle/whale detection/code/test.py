# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 14:50:43 2015

@author: weizhi
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 21:16:49 2015

@author: weizhi
"""


import pandas as pd

trainLabel = pd.read_csv('/Users/weizhi/Desktop/kaggle/whale detection/train 2.csv')
testLabel = pd.read_csv('/Users/weizhi/Desktop/kaggle/whale detection/sample_submission.csv')

import site; site.addsitedir("/usr/local/lib/python2.7/site-packages")
import cv2

img = cv2.imread('/Users/weizhi/Desktop/kaggle/whale detection/w_7489.jpg',-1)


cv2.imshow('img',img)

cv2.waitKey(0)
cv2.destroyAllWindows()

img1 = cv2.resize(img,(200,200))


cv2.imshow('img',img1)

cv2.waitKey(0)
cv2.destroyAllWindows()    

#%%
import pylab as plt
plt.figure()
plt.hist(img.ravel(),256,[0,256]); plt.show()


color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()


#%%
import numpy as np
plt.figure()
mask = np.zeros(img.shape[:2], np.uint8)
mask[1000:1600, 1000:1500] = 255
masked_img = cv2.bitwise_and(img,img,mask = mask)

# Calculate histogram with mask and without mask
# Check third argument for mask
hist_full = cv2.calcHist([img],[0],None,[256],[0,256])
hist_mask = cv2.calcHist([img],[0],mask,[256],[0,256])

plt.subplot(221), plt.imshow(img, 'gray')
plt.subplot(222), plt.imshow(mask,'gray')
plt.subplot(223), plt.imshow(masked_img, 'gray')
plt.subplot(224), plt.plot(hist_full), plt.plot(hist_mask)
plt.xlim([0,256])

plt.show()
#%%
plt.figure()

laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])

plt.show()

#%%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as nd

from skimage.util import img_as_float
from skimage.filters import gabor_kernel
import cv2
#%% reading data
import matplotlib.cm as cm
import site; site.addsitedir("/usr/local/lib/python2.7/site-packages")
import numpy as np
from numpy import random
import cv2
import pylab as plt
# -*- coding: utf-8 -*-


#%% loading each files
import glob, os
import collections
import pandas as pd

# https://docs.python.org/2/library/os.html
def findFilePath(path):
    os.chdir(path)
    filePaths = []
    for file in glob.glob("*.jpg"):
        filePaths.append(file)
    return filePaths

#%% hours and generate the outputs
# deal with each csv file 
row = img.shape[0]
col = img.shape[1]
surf = cv2.SURF()
orb = cv2.ORB()
def readCSV(path, filePaths,row,col):
   # number = 500    
    data = np.zeros([100,100,len(filePaths)])
   # kpList = []
  #  desList = []    
    for i in range(len(filePaths)):
        print i
        filePath = path + '/' + filePaths[i]
        try:
            img1 =cv2.imread(filePath,0)
            img1 = cv2.resize(img1,(100,100))

        except:
            print "wrong image"
         #   img1 = np.zeros((100,100))

        data[:,:,i] = img1
    return data #,kpList,desList

def matchesGet(desList):
    matchesList = []
    for i in range(len(desList)-1):
        matches = bf.match(desList[i],desList[i+1])
        matches = sorted(matches, key = lambda x:x.distance)
        len(matches)
        matchesList.append(matches)

    return matchesList


#img2 = cv2.drawKeypoints(img1,kp1[:10],None,(255,0,0),4)
#plt.figure()
#plt.imshow(img2),plt.show()
#%% testing
path = '/Users/weizhi/Desktop/kaggle/whale detection/imgs/'
filePaths = findFilePath(path)


#%%
data = readCSV(path , testLabel['Image'],row,col)

#data  = readCSV(path , trainLabel['Image'],row,col)
#%% get the features
fea_det = cv2.FeatureDetector_create("SIFT")
des_ext = cv2.DescriptorExtractor_create("SIFT")
des_list = []
for index in range(data.shape[-1]):
    print index
    dataUse = data[:,:,index].astype(np.uint8)
    kpts = fea_det.detect(dataUse)
    kpts, des = des_ext.compute(dataUse,kpts)
    des_list.append((testLabel['Image'][index],des))
    
#%% use the k-means
descriptors = des_list[0][1]
for img_paths, descriptor in des_list[1:]:
    try:
        descriptors = np.vstack((descriptors, descriptor))  
    except:
        print "wrong image"

from scipy.cluster.vq import *    
k = 100
voc, variance = kmeans(descriptors,k,1)

test_features = np.zeros((len(testLabel['Image']),k),'float32')
for i in range(len(trainLabel['Image'])):
    try:
        words, distance = vq(des_list[i][1],voc)
        for w in words:
            im_features[i][w] +=1
    except:
        print "error message"
#%% perform the Tf-Idf vectorization 

nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(testLabel['Image'])+1) / (1.0*nbr_occurences + 1)), 'float32')
#%% scaling the words
from sklearn.preprocessing import StandardScaler

#stdSlr = StandardScaler().fit(test_features)
test_features = stdSlr.transform(test_features)



#%%
#from sklearn.svm import LinearSVC
#from sklearn.externals import joblib
#from scipy.cluster.vq import *
#from sklearn.preprocessing import StandardScaler
#
#clf = LinearSVC()
#clf.fit(im_features, np.array(target))
#


#%%
#testData = readCSV(path , testLabel['Image'],row,col)
#target = trainLabel['whaleID']
#mathccesList = matchesGet(desList)
