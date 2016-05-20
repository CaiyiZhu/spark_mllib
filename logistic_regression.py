# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:17:31 2016

@author: csi-digital4


class pyspark.mllib.classification.LogisticRegressionWithSGD
class pyspark.mllib.classification.LogisticRegressionWithlLBFGS
"""

import numpy as np
import time


from pyspark import SparkContext, SparkConf
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint

conf = SparkConf().setAppName("kmeans clustering ").setMaster('local')
sc = SparkContext(conf=conf)
sc.setLogLevel('ERROR')

nb_train = 1000000
nb_test = nb_train / 5
P =  10
X_train = sc.parallelize(np.random.uniform(0,10,[nb_train,P]))
w = np.random.uniform(size=[P+1,])
y_train = X_train.map(f = lambda a: w[0] + np.dot(a,w[1:]))
y_train_mean = y_train.mean()
y_train = y_train.map(f = lambda val: 1 if val > y_train_mean else 0)
data_train = y_train.zip(X_train).map(f = lambda tu: LabeledPoint(tu[0],tu[1]))


X_test = sc.parallelize(np.random.uniform(0,10,[nb_test,P]))
y_test = X_test.map(f = lambda a: w[0] + np.dot(a,w[1:]))
y_test_mean = y_test.mean()
y_test = y_test.map(f = lambda val: 1 if val > y_test_mean else 0)

 
t1 = time.time()    
lrm = LogisticRegressionWithSGD.train(data_train,iterations = 10)
y_pred = lrm.predict(X_test)

print "*******************************"
nb_corr = np.sum(np.array(y_test.collect()) == np.array(y_pred.collect()))
print nb_corr
print "the accuracy is ", nb_corr/float(nb_test)
print "*******************************"
t2 = time.time()
print "time elapsed spark logistic regression ", t2 - t1



lrm_bfgs = LogisticRegressionWithLBFGS.train(data_train)
y_pred = lrm_bfgs.predict(X_test)
print "*******************************"
nb_corr = np.sum(np.array(y_test.collect()) == np.array(y_pred.collect()))
print nb_corr
print "the accuracy is ", nb_corr/float(nb_test)
print "*******************************"

t1 = time.time()
# sklearn logisicRegression model, work well
from sklearn import linear_model
lrm_sklearn = linear_model.LogisticRegression()
lrm_sklearn.fit(X_train.collect(), y_train.collect())
y_pred = lrm_sklearn.predict(X_test.collect())

print "*******************************"
nb_corr = np.sum(np.array(y_test.collect()) == y_pred)
print nb_corr
print "the accuracy is ", nb_corr/float(nb_test)
print "*******************************"

t2 = time.time()
print 'time elapsed ', t2 - t1

"""
Note:
1) The accuacy is not good, on the contrary, sklearn.logisticRegression provide very good performance
2) It seems that increasing the number of iterations increase the accuracy, but the accuracy is still not good
"""