# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 16:56:06 2022

@author: vinay
"""

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Read the data
df=pd.read_csv('news.csv')

#Get shape and head


#DataFlair - Get the labels
labels=df.label

#DataFlair - Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#DataFlair - Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#DataFlair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)
#DataFlair - Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

    
tnews="Killing Obama administration rules, dismantling Obamacare and pushing through tax reform are on the early to-do list."
#t_data=tfidf_vectorizer2.fit_transform(tnews)
x_test[1267]=tnews


#DataFlair - Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')
import streamlit as st

tfidf_test=tfidf_vectorizer.transform(x_test)
ans=pac.predict(tfidf_test)
print(ans[-1])
