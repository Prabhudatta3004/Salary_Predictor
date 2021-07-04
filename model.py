# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 17:48:13 2021

@author: prabhudatta
"""

##importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import pickle##to save model in disk

##getting the dataset
data=pd.read_csv("hiring.csv")
data.head()
##data has nan values
data["experience"].fillna(0,inplace=True)
data["test_score"].fillna(data["test_score"].mean(),inplace=True)
def to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8,
                'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0: 0}
    return word_dict[word]
data["experience"]=data["experience"].apply(lambda x:to_int(x))
##spliting the dataset
x=data.iloc[:,:3]
y=data.iloc[:,-1]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
##x_train,x_test,y_train,y_test=train_test_split(x,y,test)
#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.
regressor.fit(x,y)

##saving model to disk using pickle
pickle.dump(regressor,open("model.pkl","wb"))


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))