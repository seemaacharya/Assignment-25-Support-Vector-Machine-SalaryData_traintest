# -*- coding: utf-8 -*-
"""
Created on Thu May 27 16:08:52 2021

@author: DELL
"""

#Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns

#loading dataset
salary_train= pd.read_csv("SalaryData_Train(1).csv")
salary_test=pd.read_csv("SalaryData_Test(1).csv")
salary_train.columns
salary_test.columns

#Getting the columns(considering columns without the numerical part)
columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

#Preprocessing
from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
for i in columns:
    salary_train[i]=label_encoder.fit_transform(salary_train[i])
    salary_test[i]=label_encoder.fit_transform(salary_test[i])
    
  
#Getting x and y columns
colnames = list(salary_train.columns)
#train test split
#train_x = salary_train[colnames[0:13]]
#train_Y = salary_train[colnames[13]]
#test_x = salary_test[colnames[0:13]]
#test_y= salary_test[colnames[13]]

train_x = salary_train.iloc[0:500,0:13]
train_y = salary_train.iloc[0:500,13]
test_x = salary_test.iloc[0:300,0:13]
test_y = salary_test.iloc[0:300,13]


#Model building using SVM
from sklearn.svm import SVC
model_linear = SVC(kernel="linear")
model_linear.fit(train_x,train_y)
train_pred_linear = model_linear.predict(train_x)
test_pred_linear = model_linear.predict(test_x)
train_linear_acc = np.mean(train_pred_linear==train_y)
#train_linear_acc = 81.80%
test_linear_acc = np.mean(test_pred_linear==test_y)
#test_linear_acc = 81.67%


#kernel=poly
model_poly = SVC(kernel="poly")
model_poly.fit(train_x,train_y)
train_pred_poly = model_poly.predict(train_x)
test_pred_poly = model_poly.predict(test_x)
train_poly_acc=np.mean(train_pred_poly==train_y)
#train_poly_acc= 81.2%
test_poly_acc = np.mean(test_pred_poly==test_y)
#test_poly_acc = 80.33%


#kernel=rbf
model_rbf = SVC(kernel="rbf")
model_rbf.fit(train_x,train_y)
train_pred_rbf = model_rbf.predict(train_x)
test_pred_rbf = model_rbf.predict(test_x)
train_rbf_acc = np.mean(train_pred_rbf==train_y)
#train_rbf_acc = 81.2%
test_rbf_acc = np.mean(test_pred_rbf==test_y)
#test_rbf_acc= 80.33


#kernel=sigmoid
model_sigmoid = SVC(kernel="sigmoid")
model_sigmoid.fit(train_x,train_y)
train_pred_sigmoid = model_sigmoid.predict(train_x)
test_pred_sigmoid = model_sigmoid.predict(test_x)
train_sigmoid_acc = np.mean(train_pred_sigmoid==train_y)
#train_sigmoid_acc = 79%
test_sigmoid_acc = np.mean(test_pred_sigmoid==test_y)
#test_sigmoid_acc = 80%