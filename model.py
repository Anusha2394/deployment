# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 12:28:14 2020

@author: kasaa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("C:/Users/kasaa/Downloads/datasets/googleplaystore.csv")
df.head()
df=df.rename(columns={"Content Rating": "ContentRating"})
df.dropna(subset=["Type","ContentRating","Android Ver","Current Ver"], inplace=True)
print(df['Rating'])

df['Rating'] = df['Rating'].fillna(df.groupby('Category')['Rating'].transform('mean'))
print(df.isnull().sum())
print(df['Category'].unique().size)
plt.figure(figsize=(8,5))
df['ContentRating']=df['ContentRating'].str.replace(' ','').str.replace('+','')
sns.countplot(df['ContentRating'])
df_2 = pd.get_dummies(df, columns=['ContentRating', 'Type'], drop_first=True)

print(df_2['Installs'].max())

df_2['Installs']=df_2['Installs'].str.replace(',','').str.replace('+','').astype('int')
print(df_2.head())

plt.figure(figsize=(15,8))
sns.countplot(data=df_2,y='Category', palette="Set2")

df_corr=df_2.corr()
plt.figure(figsize=(6,6))
sns.heatmap(df_corr)

plt.figure(figsize=(15,20))
sns.countplot(data=df_2,y='Genres', palette="Set2")
print(df_2['Size'])
df_2['Size']=df_2['Size'].str.replace('M','e+6').str.replace('k','e+3').str.replace('Varies with device','0').astype('float')
df_2['Price']=df_2['Price'].str.replace('$','').astype('float')
df_2['Reviews']=df_2['Reviews'].astype('int')
df_2['Last Updated']=pd.to_datetime(df_2['Last Updated'])
df_2['before update']=df_2['Last Updated'].max()-df_2['Last Updated']
#App hasn't been updated
print(df_2[df_2['before update']==df_2['before update'].max()])
df_2 =df_2[~df_2.isin([np.nan, np.inf, -np.inf]).any(1)]
print(df_2.info())
data_model_x=df_2[['Category','Reviews','Size','Installs','Price','ContentRating_Everyone','ContentRating_Everyone10','ContentRating_Mature17','ContentRating_Teen','ContentRating_Unrated','Type_Paid']]

data_model_y=df_2[['Rating']]

# from sklearn.preprocessing import MinMaxScaler
# scalar=MinMaxScaler()
# scalar.fit(data_model_x[['before update']])
# data_model_x[['before update']]=scalar.transform(data_model_x[['before update']])
                   
print(data_model_x.info())


encoded_x=pd.get_dummies(data_model_x, columns=['Category'])


print(encoded_x.isnull().sum())


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(encoded_x,data_model_y,random_state=0)
print(x_train)
for col in x_train.columns: 
    print(col) 
print(y_train.shape,y_test.shape)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

# from sklearn.ensemble import RandomForestRegressor
# rfr = RandomForestRegressor(max_depth=8,random_state=0)
# rfr.fit(x_train,y_train)
from sklearn import svm
svm_fit=svm.SVR(C=2.0,epsilon=0.3)
svm_fit.fit(x_train,y_train)
print(x_test)
y_pred_svm=svm_fit.predict(x_test)
print(y_pred_svm)
import pickle

pickle.dump(svm_fit, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))



















