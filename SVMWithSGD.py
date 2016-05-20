# -*- coding: utf-8 -*-
"""
Created on Mon May  9 16:39:06 2016

@author: csi-digital4


SVM
"""
from __future__ import division
import numpy as np
import time

from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import SVMWithSGD
from pyspark.mllib.regression import LabeledPoint

conf = SparkConf().setAppName("kmeans clustering ").setMaster('local')
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

M = 10000
P =  10
X = np.random.uniform(0,10,[M,P])
w = np.random.uniform(size=[P+1,])
y = np.dot(X,w[1:]) + w[0]

# data shuffling
ind = range(M)
np.random.shuffle(ind) 
X = X[ind]
y = y[ind]
y = [1 if i > np.mean(y) else 0 for i in y]

X_train = X[M//5:]
y_train = y[M//5:]
X_test = X[:M//5]
y_test = y[:M//5]
# convert to mllib data format
data = []
for i in range(len(X_train)):
    data.append(LabeledPoint(y_train[i], X_train[i]))
data_train = data[M//5:]



t1 = time.time()

svm = SVMWithSGD.train(sc.parallelize(data_train))
y_pred = np.asarray(svm.predict(sc.parallelize(X_test)).collect())





print "*******************************"
#print len(y_pred), len(y_test)
print "model using spark.mllib"
print np.sum(y_test == y_pred)
print "the accuracy is ", np.sum(y_test == y_pred)/len(y_test)

t2 = time.time()
print "time elapsed", t2 - t1
print "*******************************"









t1 = time.time()
# sklearn logisicRegression model, work well
from sklearn.svm import SVC
svc_sklearn = SVC()
svc_sklearn.fit(X_train, y_train)
y_pred = svc_sklearn.predict(X_test)

print "*******************************"
print "model using sklearn"
print np.sum(y_test == np.array(y_pred))
print "the accuracy is ", np.sum(y_test == y_pred)/len(y_test)

t2 = time.time()
print 'time elapsed ', t2 - t1
print "*******************************"




"""

"""


