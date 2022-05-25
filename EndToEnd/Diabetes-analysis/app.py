#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 19:34:11 2022

@author: sparshjhariya
"""

from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"



    
    
@app.route('/predict',methods=["GET"])
def predict_function():
    """Predict whether a person is diabetic or not
    ---
    parameters:  
      - name: age
        in: query
        type: float
        required: true
      - name: sex
        in: query
        type: float
        required: true
      - name: bmi
        in: query
        type: float
        required: true
      - name: bp
        in: query
        type: float
        required: true
      - name: s1
        in: query
        type: float
        required: true
      - name: s2
        in: query
        type: float
        required: true
      - name: s3
        in: query
        type: float
        required: true
      - name: s4
        in: query
        type: float
        required: true
      - name: s5
        in: query
        type: float
        required: true
      - name: s6
        in: query
        type: float
        required: true
    responses:
        200:
            description: The output values
        
    """
    
    age=request.args.get("age")
    sex=request.args.get("sex")
    bmi=request.args.get("bmi")
    bp=request.args.get("bp")
    s1=request.args.get("s1")
    s2=request.args.get("s2")
    s3=request.args.get("s3")
    s4=request.args.get("s4")
    s5=request.args.get("s5")
    s6=request.args.get("s6")
    a = [age,sex,bmi,bp,s1,s2,s3,s4,s5,s6]
    b = np.array(a, dtype=float)
    prediction=classifier.predict([b])
    print(prediction)
    return "The predicted values is: " +str(prediction)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)