# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:20:17 2020

@author: Mukund Rastogi
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

    
@app.route('/predict',methods=['POST'])

def predict():
    '''
    For rendering results on HTML GUI
    '''
    def R(recency):
        if recency<=17:
            return 1
        elif recency<=50:
            return 2
        elif recency<=142:
            return 3
        else:
            return 4
    
    def F(frequency):
        if frequency<=17:
            return 4
        elif frequency<=41:
            return 3
        elif frequency<=99:
            return 2
        else:
            return 1

    def M(monetary):
        if monetary<=300:
            return 4
        elif monetary<=652:
            return 3
        elif monetary<=1576:
            return 2
        else:
            return 1

    int_features = [int(x) for x in request.form.values()]
    r=R(int_features[0])
    f=F(int_features[1])
    m=M(int_features[2])
    score=r+f+m
    int_features.extend([r,f,m,score])
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = prediction[0]
    if output==1:
        result="the best"
    elif output==2:
        result="most loyal"
    elif output==3:
        result="almost lost"
    else:
        result="lost"

    return render_template('index.html', prediction_text='Customer is {}'.format(result))




if __name__ == "__main__":
    app.run(debug=True)