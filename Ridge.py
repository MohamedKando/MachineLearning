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
import  re

data=pd.read_csv('cars-train.csv')
data=data.dropna(how='any',axis=0)
data2=featureClean(data)

data3,X,Y=featureClean2(data2)

#Feature Selection
#Get the correlation between the features
corr = data3.corr()
#Top 50% Correlation training features with the Value
top_feature = corr.index[abs(corr['price(USD)']) > 0.2]
#Correlation plot

top_corr = data3[top_feature].corr()

top_feature = top_feature.delete(-1)

X = X[top_feature]

#Split the data to training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y,
            test_size = 0.30,shuffle=True,random_state=42)


from sklearn.linear_model import RidgeCV
model = RidgeCV(alphas=range(1, 100))
model.fit(X_train, y_train)
prediction = model.predict(X_test)





# Performance
print("\t~~~~~ Ridge Regressor ~~~~~")
print('Mean Square Error: ', metrics.mean_squared_error(np.asarray(y_test), prediction))
print("Model Accuracy(%): \t" + str(r2_score(y_test, prediction)*100) + "%")
print("~~~~  ~~~~~  ~~~~~  ~~~~~  ~~~~~  ~~~~~  ~~~~~  ~~~~~ ")
