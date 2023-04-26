#Importing Libraries 

import pandas as pd
import numpy as np
import re          #regular expression
#import seaborn as sns
#import matplotlib.pyplot as plt
import warnings
warnings.simplefilter("ignore")

# Read data
df = pd.read_csv("Language Detection.csv")

# feature and label extraction
X = df["Text"]
y = df["Language"]

# Label encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# cleaning the data
text_list = []

# iterating through all the text
for text in X:         
    text = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', text) # removes all the symbols and numbers
    text = re.sub(r'[[]]', ' ', text)   
    text = text.lower()          # converts all the text to lower case
    text_list.append(text)       # appends the text to the text_list
    
    
# Encode the feature(text)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer() 
X = cv.fit_transform(text_list).toarray() # tokenize a collection of text documents and store it
                                            #in an array
                                            
# split the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=32)

# Model Training
from sklearn.naive_bayes import MultinomialNB  
model = MultinomialNB()
model.fit(x_train, y_train)

# prediction
y_pred = model.predict(x_test)

# model evaluation
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("Accuracy is :",accuracy)

def predict(text):
     x = cv.transform([text]).toarray() # convert text to bag of words model (Vector)
     language = model.predict(x) # predict the language
     lang = le.inverse_transform(language) # find the language corresponding with the predicted value
     print("The language is in", lang[0]) # printing the language
     
predict("I'm a boy")

