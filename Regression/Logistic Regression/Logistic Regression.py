#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Imports here
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data reads are done here
train_data = pd.read_csv("train-data.csv")
test_data = pd.read_csv("test-data.csv")

# Print Stats of the data
print(train_data.head())
print("Shape of the train_data is : ", train_data.shape)
print("Shape of the test_data is  : ", test_data.shape)

# Initializing train x and y
train_x = train_data.drop(columns = ['Survived'], axis = 1)
train_y = train_data['Survived']

#Initializing test x and y
test_x = test_data.drop(columns = ['Survived'], axis = 1)
test_y = test_data['Survived']


# Creating the instance of the model
model = LogisticRegression(solver='lbfgs', max_iter=1000)

# Fit the model with the training data 
model.fit(train_x,train_y)
 
# Calculate the coeffecient of the model
print("********************************************************************")
print("Coeffecient of the model   \n: ", model.coef_)

# Calculate the intercept of the model
print("********************************************************************")
print("Intercept of the model     \n: ", model.intercept_)

# Train the model with the training data set
predict_train = model.predict(train_x)
print("********************************************************************")
print("Target on Train Data       \n: ", predict_train)

accuracy_train = accuracy_score(train_y, predict_train)
print("********************************************************************")
print("Accuracy on the Train Data \n: ",accuracy_train)

# Predict the target on the Test Data
predict_test = model.predict(test_x)
print("********************************************************************")
print("Target on Test Data        \n:", predict_test)

# Accuracy score on the Test Data
accuracy_test = accuracy_score(test_y, predict_test)
print("********************************************************************")
print("Accuracy Score on Test Dataset : ", accuracy_test)
