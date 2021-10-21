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

os.chdir('C://Code/Kaggle/heart-failure/')

# Headers
st.title('Heart Failure Classification')
st.image('heart-black.jpg')
st.write('Photo by [Alexandru Acea](https://unsplash.com/@alexacea?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)')

# EDA
st.write('We used the heart failure prediction [dataset](https://www.kaggle.com/fedesoriano/heart-failure-prediction).')

# User input
sex_button = st.sidebar.radio('Gender',('Male','Female'))
age_slider = st.sidebar.slider('Age',20,100,60)
bp_slider = st.sidebar.slider('Resting blood pressure',100,200,150)
chol_slider = st.sidebar.slider('Serum cholesterol',200,300,250)

# Load trained model and scaler
scaler = pickle.load(open('std_scaler.sav','rb'))
model = pickle.load(open('cls_model.sav','rb'))

# Prep data
if sex_button == 'Male':
    input_sex = 1
else:
    input_sex = 0
    
# Record = Age, resting BP, cholesterol, sex
input_rec = np.array([age_slider,bp_slider,chol_slider,input_sex])
input_rec_std = scaler.transform(input_rec)

probs_rec = model.predict_proba(input_rec_std)

st.write(f'The predicted probability of heart disease is {probs_rec/100:8.2f}%')
# Diagnostics - AUC diagram