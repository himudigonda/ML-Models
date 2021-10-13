#!/usr/bin/env python
# -*- coding: utf-8 -*-
#! @author : @ruhend (Mudigonda Himansh)

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

train_data = pd.read_csv('./train-data.csv')
test_data = pd.read_csv('./test-data.csv')

print('Shape of the train dataset  : ', train_data.shape)
print('Shape of the test dataset   : ', test_data.shape)

train_x = train_data.drop(columns=['Survived'], axis=1)
train_y = train_data['Survived']

test_x = test_data.drop(columns=['Survived'], axis=1)
test_y = test_data['Survived']

model = KNeighborsClassifier()

model.fit(train_x, train_y)

print('The number of neighbors used to predict the target : ',
      model.n_neighbors)

predict_train = model.predict(train_x)
print('Target on train data          : ', predict_train)
