#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports here
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Read the train and test datasets
test_data, train_data = pd.read_csv('train-data.csv'), pd.read_csv(
    'test-data.csv')
print(test_data.head())
print(train_data.head())

# Stats of the Dataset
print("Shape of the training data: ", train_data.shape)
print("Shape of the testing Data : ", test_data.shape)

# Initializing the train_x, train_y, test_x and test_y values from the given Dataset
train_x = train_data.drop(columns=['Survived'], axis=1)
train_y = train_data['Survived']
test_x = test_data.drop(columns=['Survived'], axis=1)
test_y = test_data['Survived']

# Creating the Instance of the model as DecisionTreeClassifier
model = DecisionTreeClassifier()

#Fit the model with the Training Data
model.fit(train_x, train_y)

print("********************************************************************")
# Here we are printing the depth of the decision tree
print("Depth of the Decision Tree : ", model.get_depth())

print("********************************************************************")
# Here we test the model with the Train Dataset
predict_train = model.predict(train_x)
print("Target on the Train Dataset  : ", predict_train)

print("********************************************************************")
# Here we find the accuracy score of the Train Dataset
accuracy_train = accuracy_score(train_y, predict_train)
print("Accuracy of the Train Dataset : ", accuracy_train)

print("********************************************************************")
# Here we test the model with the Test Dataset
predict_test = model.predict(test_x)
print("Target on the Test Datset : ", predict_test)

print("********************************************************************")
# Here we find the accuracy of the model using the Test Dataset
accuracy_test = accuracy_score(test_y, predict_test)
print("Accuracy of the Test Dataset : ", accuracy_test)
