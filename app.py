# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 16:59:07 2020

@author: Meliza
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 15:33:42 2020

@author: Meliza
"""

import numpy as np
import pickle
from flask import Flask, render_template, jsonify, request

# Initializing the Flask Application
app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))


#Routing the application to root folder
@app.route('/')

def home():
    return render_template('index.html')

#Routing to the prediction outcome
@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_data = [np.array(int_features)]
    prediction = model.predict_proba(final_data)
    output = round(prediction[0][0]*100,2) 
    
    return render_template('index.html', prediction_text='Prabablity of buying {}%'.format(output))
    #return render_template('index.html', prediction_text='Hello')

if __name__ == '__main__' :
    app.run(debug=True)                           