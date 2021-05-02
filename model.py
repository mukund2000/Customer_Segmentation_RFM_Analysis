# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 11:19:34 2020

@author: Mukund Rastogi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('Customer_Segmentation.csv')
data.head()
# drop insignificant data
RFM_data=data.drop(['CustomerID','RFMGroup'],axis=1)
# Encoding Categorical columns into Numeric Features
RFM_data.loc[RFM_data['RFM_Loyalty_Level']=='Platinum','RFM_Loyalty_Level']=1
RFM_data.loc[RFM_data['RFM_Loyalty_Level']=='Gold','RFM_Loyalty_Level']=2
RFM_data.loc[RFM_data['RFM_Loyalty_Level']=='Silver','RFM_Loyalty_Level']=3
RFM_data.loc[RFM_data['RFM_Loyalty_Level']=='Bronze','RFM_Loyalty_Level']=4

RFM_data['Monetary']=RFM_data['Monetary'].astype(int)
# checking for null values
RFM_data.isnull()
#checking for data types of each Feature
RFM_data.info()

import seaborn as sns
corr = RFM_data.corr(method='kendall')
plt.figure(figsize=(20,10))
sns.heatmap(corr, annot=True)

RFM_data.groupby('RFM_Loyalty_Level')['R'].count()

y=RFM_data['RFM_Loyalty_Level']
x=RFM_data.drop(['RFM_Loyalty_Level'],axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

from sklearn.ensemble import RandomForestClassifier #Random Forest
from sklearn.neighbors import KNeighborsClassifier #KNN
from sklearn.naive_bayes import GaussianNB #Naive bayes
from sklearn.tree import DecisionTreeClassifier #Decision Tree
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_RF = randomforest.predict(x_test)
score_randomforest = randomforest.score(x_test,y_test)
print('The accuracy of the Random Forest Model is', score_randomforest)

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_GNB = gaussian.predict(x_test)
score_gaussian = gaussian.score(x_test,y_test)
print('The accuracy of Gaussian Naive Bayes is', score_gaussian)

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_LR = logreg.predict(x_test)
score_logreg = logreg.score(x_test,y_test)
print('The accuracy of the Logistic Regression is', score_logreg)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_KNN = knn.predict(x_test)
score_knn = knn.score(x_test,y_test)
print('The accuracy of the KNN Model is',score_knn)

cfm=confusion_matrix(y_test, y_GNB)
sns.heatmap(cfm, annot=True)
plt.ylabel('Predicted classes')
plt.xlabel('Actual classes')

import pickle
# Saving model to disk
pickle.dump(randomforest, open('model.pkl','wb'))
#print(xTest.info())
#print(yTest.info())
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
model.score(x_test,y_test)
print(model.predict([[8,115,6196,1,1,1,3]]))
