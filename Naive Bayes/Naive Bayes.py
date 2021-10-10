#!/usr/bin/env python
# -*- coding: utf-8 -*-
#! @author : @ruhend (Mudigonda Himansh)

# imports here
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# import datasets
train_data = pd.read_csv('./train-data.csv')
test_data = pd.read_csv('./test-data.csv')

# print stats of the datasets
print('Shape fo the training dataset    : ', train_data.shape)
print('Shape of the testing dataset     : ', test_data.shape)
print('Head of the training dataset     : ', train_data.head(5))
print('Head of the testing dataset      : ', test_data.head(5))

# split the train dataset in the respective axes
train_x = train_data.drop(columns=['Survived'], axis=1)
train_y = train_data['Survived']

# split the test dataset in the respective axes
test_x = test_data.drop(columns=['Survived'], axis=1)
test_y = test_data['Survived']

# Create an instance of the Naives Bayes ML Model
model = GaussianNB()

# Fit the model with the dataset
model.fit(train_x, train_y)

# Make a prediction model on the train dataset
predict_train = model.predict(train_x)
print('Target on the train dataset       : ', predict_train)

# Find the accuracy of the created model w.r.t train dataset
accuracy_train = accuracy_score(train_y, predict_train)
print('Accuracy on the training dataset : ', accuracy_train)

# Now make predictions on the test dataset
predict_test = model.predict(test_x)
print('Target on the test dataset       : ', predict_test)

# Find the accuracy of the created model w.r.t train dataset
accuracy_test = accuracy_score(test_y, predict_test)
print('Accuracy on the training dataset : ', accuracy_test)
