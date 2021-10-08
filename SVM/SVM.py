#!/usr/bin/env python
# -*- coding: utf-8 -*-
#! @author: @ruhend

# Imports here
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('./train-data.csv')
test_data = pd.read_csv('./test-data.csv')

# Print Stats of the data sets
print(train_data.head())
print(train_data.shape)
print(test_data.shape)

# seperate independent variable from the train dataset
train_x = train_data.drop(columns=['Survived'], axis=1)
train_y = train_data['Survived']

# seperate independent variable from the test dataset
test_x = test_data.drop(columns=['Survived'], axis=1)
test_y = test_data['Survived']

# make an instance of the SVC model
model = SVC()

# fit the dataset into the model
model.fit(train_x, train_y)

# print the target of the model with train data
predict_train = model.predict(train_x)
print("The target on the dataset is : ", predict_train)

# print the accuracy_score of the model with the train dataset
accuracy_train = accuracy_score(train_y, predict_train)
print("Accuracy of the train dataset is : ", accuracy_train)

#print the target of the model with test_data
predict_test = model.predict(test_x)
print("The target on the test dataset is : ", predict_test)

# print the accuracy_score of the model with test_data
accuracy_test = accuracy_score(test_y, predict_test)
print("Accuracy of the test dataset is : ", accuracy_test)
