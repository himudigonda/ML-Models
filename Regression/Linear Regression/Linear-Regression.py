
# @author: @ruhend (Mudigonda Himansh)
# Linear Regression


import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# reading the train and test data sets
train_data, test_data = pd.read_csv('train.csv'), pd.read_csv('test.csv')

# printing out train and test data sets and their 
print(train_data.head())
print(test_data.head())
print("Shape of training data: ", train_data.shape)
print("Shape of testing data: ", test_data.shape)

# Drop the Item_Outlet_sales from the x axis to shift that to y axis as that is the one we would be recording on
train_x = train_data.drop(columns=['Item_Outlet_Sales'],axis=1)
train_y = train_data['Item_Outlet_Sales']
test_x = test_data.drop(columns=['Item_Outlet_Sales'],axis=1)
test_y = test_data['Item_Outlet_Sales']

# Creating the instance of the Linear Regression ML Model!
model = LinearRegression()

# Training the Linear Regression Model
model.fit(train_x, train_y)

# Printing the Stats from the Trained Model
print()
print("***********************************")
print("The coeffecient of the model             \n: ", model.coef_)

print()
print("***********************************")
print("The intercept of the model              \n : ", model.intercept_)

# Now let us train the model using train dataset
predict_train = model.predict(train_x)
print("***********************************")
print("Item_Outlet_Sales on the training data  : \n",predict_train)
# Now let us calculate the RMSE of the train dataset
rmse_train = mean_squared_error(train_y,predict_train)**(0.5)
print("RMSE of the train dataset                : ",rmse_train)

# Now let us test the trained model
predict_test = model.predict(test_x)
print("***********************************")
print("Item_Outlet_Sales on the training data  : \n",predict_test)
# Now let us calculate the RMSE of the train dataset
rmse_test = mean_squared_error(test_y,predict_test)**(0.5)
print("RMSE of the train dataset                : ",rmse_test)


