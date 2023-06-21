import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder,PolynomialFeatures
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import metrics
from model import *
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import  re

data=pd.read_csv('cars-train.csv')
data2=featureClean(data)
data3,X,Y=featureClean2(data2)

#Feature Selection
#Get the correlation between the features

data3,X,Y=featureClean2(data2)
corr = data3.corr()

#Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['price(USD)']) > 0.2]


top_corr = data3[top_feature].corr()

top_feature = top_feature.delete(-1)

X = X[top_feature]



#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y,
            test_size = 0.10,shuffle=False,random_state=42)


model = RandomForestRegressor()     # Default parameters are fine
model.fit(X_train, y_train)
prediction = model.predict(X_test)


# Performance
print("~~~~~ Random Forest Regressor ~~~~~")
print('Mean Square Error: ', metrics.mean_squared_error(np.asarray(y_test), prediction))
print("Model Accuracy(%): \t" + str(r2_score(y_test, prediction)*100) + "%")
print("~~~~  ~~~~~  ~~~~~  ~~~~~  ~~~~~  ~~~~~  ~~~~~  ~~~~~ ")
