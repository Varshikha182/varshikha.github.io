import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import re

cv = CountVectorizer()
le = LabelEncoder()

app = Flask(__name__)
#model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    
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

    if request.method == 'POST':
        txt = request.form['text']
        t_o_b = cv.transform([txt]).toarray()# convert text to bag of words model (Vector)
        language = model.predict(t_o_b) # predict the language
        corr_language = le.inverse_transform(language) # find the language corresponding with the predicted value
    
        output = corr_language[0]

    return render_template('index.html', prediction='Language is in {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)