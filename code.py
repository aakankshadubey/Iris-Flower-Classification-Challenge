#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 08:32:05 2020
Iris Flower Classification Challenge Using K-Nearest Neighbors (KNN)
@author: Aakanksha Dubey
"""


# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Importing Dataset
from sklearn.datasets import load_iris
dataset = load_iris()

# Print Details of Dataset
print("Shape of Data: {}".format(dataset.data.shape))
print(dataset.feature_names)
print(dataset.target_names)
print(dataset.data.shape)
print(dataset.target.shape)
print(dataset.data)
print(dataset.target)

# Creating Dataframe
X = pd.DataFrame(dataset.data,columns =  dataset.feature_names)
y = pd.DataFrame(dataset.target)
print(X.shape)
print(y.shape)


# Splitting Data into Training Set and Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# Fitting KNN to Training Set
classifier = KNeighborsClassifier(n_neighbors = 5, 
                                  metric = "minkowski",
                                  p = 2)

classifier.fit(X_train, y_train.values.ravel())


# Predicting Results of Test Set
y_pred = classifier.predict(X_test)


# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Test set score (KNN.score): {:.4f}".format(classifier.score(X_test, y_test)))
diagonal_sum = cm.trace()
total_sum = cm.sum()
print("Accuracy from Confusion Matrix : {:.4f}".format(diagonal_sum / total_sum))

