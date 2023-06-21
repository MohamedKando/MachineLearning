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

data=pd.read_csv('cars-train.csv')

def Feature_Encoder(x,cols):
    for i in cols:
        lb=LabelEncoder()
        lb.fit(list(x[i].values))
        x[i]=lb.transform(list(x[i].values))
    return x
def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X

def featureClean (data):

    data['fuel_type'] = data['fuel_type'].str.lower()
    data[['model','car','year']] = data['car-info'].str.split(",", expand=True)
    data['model']=data['model'].str.replace('[','',regex=True).str.replace(']','',regex=True).str.replace('(','',regex=True).str.replace(')','',regex=True)
    data['car']=data['car'].str.replace('[','',regex=True).str.replace(']','',regex=True).str.replace('(','',regex=True).str.replace(')','',regex=True)
    data['year']=data['year'].str.replace('[','',regex=True).str.replace(']','',regex=True).str.replace('(','',regex=True).str.replace(')','',regex=True)
    data=data.drop(columns=['car-info'],axis=1)
    data = data.drop(columns=['color'], axis=1)
    cols = list(data.columns.values)  # Make a list of all of the columns in the df
    cols.pop(cols.index('price(USD)'))  # Remove b from list
    cols.pop(cols.index('year'))  # Remove x from list
    data = data[cols + ['year', 'price(USD)']]  # Create new dataframe with columns in the order you want
    data['year'] = data['year'].astype(int)
    return data



def featureClean2(data2):


    #Feature Encoding

    data2['segment'].fillna((data2['segment'].mode().iloc[0]), inplace=True)
    cols = ('segment', 'fuel_type', 'condition', 'transmission', 'drive_unit', 'model', 'car', 'year')
    data2 = Feature_Encoder(data2, cols)
    data2['volume(cm3)'].fillna((data2['volume(cm3)'].mean()), inplace=True)
    data2['drive_unit'].fillna((data2['drive_unit'].mode().iloc[0]), inplace=True)
    X = data2.iloc[:, 1:11]
    Y = data2['price(USD)']
    X=pd.DataFrame(X)



    return data2,X,Y

data2=featureClean(data)

data3,X,Y=featureClean2(data2)


