# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 23:28:59 2021

@author: ntuse

"""
"""Task:1 Importing the required packages"""

# importing the required packages for the simple classification

# to work with dataframes we import pandas
import pandas as pd

# to work with matrices we import numpy
import numpy as np

#to work with visualiztion plots we import  seaborn
import seaborn as sns

#to partition the data
from sklearn.model_selection import train_test_split

# to get the Logistic Regression
from sklearn.linear_model import LogisticRegression

#importing the perfomance metrices,accuracyscore
from sklearn.metrics import accuracy_score,confusion_matrix

"""Step:2 - Importing the Data"""
data_income = pd.read_csv("D:\study&work\income(1).csv")

# creating a copy of data
data=data_income.copy()

"""Task2: 
    #Exploring the data analysis:
    ->1.Getting to know the data
    ->2.Data preprocessing(Missing values)
    ->3.Cross tables and data visualization

"""
## getting to know the data
# To chech the variable datatypes

print(data.info())


# To check the missing values we use the method isnull() or isna()

data.isnull()

# to get the no. of  missing values in the attribute or column

print("Data columns with null values:'n",data.isnull().sum())

# @@@@ no missing values!

#!!!!! Summary of numerical variables

summary_num = data.describe()
print(summary_num)

# Summary of categorical values
summary_cat=data.describe(include="O")
print(summary_cat)

#Frequency of each categories

data['JobType'].value_counts()
data['occupation'].value_counts()

# checking for unique classes and unique is a numpy method
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))

"""
Go back and read the data by including "na_values['?']
"""
data =pd.read_csv("D:\study&work\income(1).csv",na_values=[" ?"])

######################################################
# Data preprocessing
######################################################
print(data.isnull().sum())

missing =data[data.isnull().any(axis=1)]
#axis => to consider atleast one column value is missing

"""
Points to note:
    1.missing values in Job type =1809
    2.Missing values in Occupation=1816
    3.There are 1809 rows where two specific columns i.e., occupation & jobtype have missing values
    4.(1816-1809) =7 => You still have occupation unfilled for these 7 rows.because jobtype is never worked
    """
data1=data.dropna(axis=0)

# to see the relationship of independent variables
correlation_val=data1.corr()

######################################################
# Cross tables & Data Visualization
#####################################################

# Exctracting the column names:
data1.columns


#####################3
# Gender Proportion table
gender=pd.crosstab(index=data1['gender'],columns='count',normalize=True)
print(gender)    

#####checking gender to salary relationship

gender_sal=pd.crosstab(index=data1['gender'],columns=data1['SalStat'],margins=True,normalize='index')
print(gender_sal)

##################################################################
#Frequency distribution of Salary status'
#################################################################
SalStat=sns.countplot(data1['SalStat'])


"""
75% of people's salary status is <-50,000
& 25% of people's salary status is >50,000
"""

###############################################################
sns.distplot(data1['age'],bins=10,kde=False)
## it seems that people with age group 20-40 are in more
sns.boxplot('SalStat','age',data=data1)
data1.groupby('SalStat')['age'].median()
##### people with age group 35-50 are likely to earn >50000
#####people with age group 25-45 are likely to earn<= 50000

#########################################################################3
##LOGISTIC REGRESSION
##########################################################################33

#Reindexing the salary status names to 0,1

data1=data.dropna(axis=0)
data1['SalStat']=data1['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data1['SalStat'])

new_data=pd.get_dummies(data1,drop_first=True)

# storting the columnn values
column_list=list(new_data.columns)
print(column_list)

#Seperating the input names from data
features=list(set(column_list)-set(['SalStat']))
print(features)

#Storing the output values iun y
y=new_data['SalStat'].values
print(y)


# storing the values from input features
x=new_data[features].values
print(x)

# Splitting the data into train and test
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

# make an instance of the model
logistic=LogisticRegression()

# Fitting the values for x and y
logistic.fit(train_x,train_y)
logistic.coef_
logistic.intercept_

# prediction from test data
prediction = logistic.predict(test_x)
print(prediction)

# Evaluating the model through confusion matrix

confusion_matrix=confusion_matrix(test_y,prediction)
print(confusion_matrix)

# calculating the accuracy
from sklearn.metrics import accuracy_score,confusion_matrix

accuracy_score=accuracy_score(test_y, prediction)
print(accuracy_score)

#Printing the misclassified values from prediction

print('Misclassified samples:%d' % (test_y != prediction).sum())

#########################################################
#LOGISTIC REGRESSION - REMOVING INSIGNIFICANT VARIABLES
###########################################################3

# Reindexing the salary status names to 0,1
data1=data.dropna(axis=0)

data1['SalStat']=data1['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data1['SalStat'])

cols = ['gender','nativecountry','race','JobType']
new_data=data1.drop(cols,axis=1)

new_data=pd.get_dummies(new_data,drop_first=True)
print(new_data.isnull().sum())
#storing the column names
column_list=list(new_data.columns)
print(column_list)

#seperating the input names from data
features=list(set(column_list)-set(['SalStat']))
print(features)
#storing the output values in y
y=new_data['SalStat'].values
print(y)

# storing the values from input features
x=new_data[features].values
print(x)

#Splitting the data into train and test
train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)

#make an instance of the model
logistic=LogisticRegression()

logistic.fit(train_x,train_y)

#Prediction from test data
prediction=logistic.predict(test_x)
print(prediction)

from sklearn.metrics import accuracy_score,confusion_matrix

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

# printing the misclassified values from prediction
print('Misclassified samples:%d'%(test_y != prediction).sum())

#############################################################33
#################################################
#######################################
##KNN

from sklearn.metrics import accuracy_score,confusion_matrix

#importing the library of KNN
from sklearn.neighbors import KNeighborsClassifier

#import library for plotting
import matplotlib.pyplot as plt

#Storing the k neaserest neighbors classifier
KNN_classifier=KNeighborsClassifier(n_neighbors=5)

#fitting the knearest neighbors classifier
KNN_classifier.fit(train_x,train_y)

#Prediciting the test values with model
prediction =KNN_classifier.predict(test_x)

#performance metric check
confusion_matrix=confusion_matrix(test_y,prediction)
print("\t","Predicted values")
print("Original values","\n",confusion_matrix)


#calculating the accuracy
from sklearn.metrics import accuracy_score,confusion_matrix

accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)

print('Misclassified samples:%d'%(test_y!=prediction).sum())

"""
Effect of k value on classifier
"""
Misclassified_sample=[]
# calculating the error for k values
for i in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pred_i=knn.predict(test_x)
    Misclassified_sample.append((test_y!=pred_i).sum())
    
print(Misclassified_sample)
 
####END SCRIPT