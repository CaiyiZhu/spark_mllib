# -*- coding: utf-8 -*-
"""
Created on Wed May  4 17:40:50 2016

@author: csi-digital4


This script implement the kmean clustering method using spark machine learning package. 
"""
import numpy as np

from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans


conf = SparkConf().setAppName("kmeans clustering ").setMaster('local')
sc = SparkContext(conf=conf)



M = 10000
P = 10
X = np.random.randn(M,P) # 10000 sample 10-D points
y = np.zeros([M,])
X[:M/2]  = X[:M/2] + 0
X[M/2:] = X[M/2:] + 10
y[M/2:] = y[M/2:] + 1 

ind = range(M)
np.random.shuffle(ind)
X = X[ind]
y = y[ind]
X_train = X[:int(M*0.8)]

X_test = X[int(M*0.8):]
y_test = y[int(M*0.8):]

# y = np.zeros([50,]) + np.ones([50,])
model = KMeans.train(sc.parallelize(X_train), 2, maxIterations=10, runs=30, initializationMode="random", seed=50, initializationSteps=5, epsilon=1e-4)
y_pred = model.predict(sc.parallelize(X_test)).collect()  

print "##############################"
print "clustering accuracy", sum(y_test==np.array(y_pred))/float((0.2*M))
print "##############################"



#import time
#t1 = time.time()
#### using sklearn
#from sklearn.cluster import KMeans
#km = KMeans(n_clusters= 2, init='k-means++', max_iter=100, n_init=1)
#km.fit(X_train)
#y_pred = km.predict(X_test)
#
#print "clustering accuracy", sum(y_test==np.array(y_pred))/float((0.2*M))
#
#t2 = time.time()
#print "time elapsed", t2-t1


