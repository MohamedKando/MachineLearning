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
import  re
from sklearn.metrics import r2_score
from model import *

data=pd.read_csv('cars-train.csv')

data2=featureClean(data)

data3,X,Y=featureClean2(data2)

#Feature Selection
#Get the correlation between the features

#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y,
            test_size = 0.25,shuffle=False,random_state=42)

poly_features = PolynomialFeatures(degree=3)
# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X_train)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
poly_model.fit(X_train_poly, y_train)

# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)

# predicting on test data-set
prediction = poly_model.predict(
            poly_features.fit_transform(X_test))


print("~ Polonomyal Regressor ~")
print('Mean Square Error: ', metrics.mean_squared_error(np.asarray(y_test), prediction))
print("Model Accuracy(%): \t" + str(r2_score(y_test, prediction)*100) + "%")
print("~~  ~  ~  ~  ~  ~  ~  ~~~ ")