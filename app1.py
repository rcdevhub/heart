# -*- coding: utf-8 -*-
"""
Streamlit app for Heart Failure classification

Created on Wed Oct 20 08:46:55 2021

@author: rcpc4
"""

# Setup
import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (f1_score,classification_report,confusion_matrix,
                             roc_auc_score,roc_curve)

# Headers
st.title('Heart Disease Classification')
st.image('heart-black.jpg')
st.write('Photo by [Alexandru Acea](https://unsplash.com/@alexacea?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)')

# User input
st.sidebar.subheader('Heart Disease Calculator')
sex_button = st.sidebar.radio('Gender',('Male','Female'))
age_slider = st.sidebar.slider('Age',20,100,60)
bp_slider = st.sidebar.slider('Resting blood pressure',100,200,150)
chol_slider = st.sidebar.slider('Serum cholesterol',100,350,600)

# Load trained models, results
scaler = pickle.load(open('std_scaler.sav','rb'))
model = pickle.load(open('cls_model.sav','rb'))
results = pickle.load(open('output/model_results.sav','rb'))

# Prep data
if sex_button == 'Male':
    input_sex = 1
else:
    input_sex = 0
    
# Record = Age, resting BP, cholesterol, sex
input_rec = np.array([age_slider,bp_slider,chol_slider,input_sex]).reshape(1,-1)
input_rec_std = scaler.transform(input_rec)

probs_rec = model.predict_proba(input_rec_std)

st.sidebar.metric('Heart Disease Probability',f'{probs_rec[0,1]*100:3.1f}%')

# Data distributions
st.subheader('Training data')
st.write('We used the heart failure prediction [dataset](https://www.kaggle.com/fedesoriano/heart-failure-prediction).')
st.image('plots/pairplot_modelled.png')
st.subheader('Model diagnostics')
st.write('Confusion matrix')
results['conf']
st.write('Classification report')
st.code(results['rep'])
st.write(f"The model AUC is {results['auc']:3.2f}")
st.image('plots/roc.png')
