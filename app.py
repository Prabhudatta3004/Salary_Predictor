# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 18:04:01 2021

@author: prabhu
"""

import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__,template_folder="templates")##creating the flask web app
model=pickle.load(open("model.pkl","rb"))##getting the model
 
@app.route("/")
def home():
     return render_template("index.html")
 ##getting the template
 
@app.route("/predict",methods=["POST"])
def predict():
     
     #For rendering results on html gui
     
     int_features=[int(x) for x in request.form.values()]
     ##to get the values as input
     final_features=[np.array(int_features)]#converting into array
     prediction=model.predict(final_features)##prediction
     
     
     output=round(prediction[0],2)
     
     return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))
 
if __name__ == "__main__":
    app.run(debug=True)