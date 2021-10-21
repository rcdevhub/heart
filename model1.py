# -*- coding: utf-8 -*-
"""
Heart failure model

Created on Mon Oct 18 12:12:50 2021

@author: rcpc4
"""

''' --------------------------Setup--------------------------'''

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (f1_score,confusion_matrix,classification_report,
                             roc_auc_score,roc_curve)

os.chdir('C://Code/Kaggle/heart-failure')

''' --------------------------Import Data--------------------'''

train_raw = pd.read_csv('data/heart.csv')

''' --------------------------EDA functions--------------------'''

def freq_by_var_bin(data,target,var,bins):
    '''Return df of target frequency by variable binned.'''
    var_bin = pd.cut(data[var],bins)
    grp = data.groupby(var_bin)
    sum_by_var = grp[target].sum().rename('sum')
    count_by_var = grp[target].count().rename('count')
    freq_by_var = (sum_by_var/count_by_var).rename('frequency')
    freq_by_var = pd.concat([sum_by_var,
                             count_by_var,
                             freq_by_var],
                             axis=1)
    
    return freq_by_var
    
''' --------------------------EDA--------------------'''

# Summary of numeric variables
desc_train = train_raw.describe()

# Categorical variable counts
cat_vars = ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']
val_counts = {}
for c in cat_vars:
    count_num = train_raw[c].value_counts().rename('count')
    count_prop = train_raw[c].value_counts(normalize=True).rename('prop')
    val_counts[c] = pd.concat([count_num,
                              count_prop],
                              axis=1)

# Heart Disease prevalence by variable (one-way)
freq_vars = ['Age','RestingBP','Cholesterol']
freqs = {}
for item in freq_vars:
    freqs[item] = freq_by_var_bin(train_raw,target='HeartDisease',var=item,bins=6)
    
# Predictor variables pairplot

plt.figure()
# plt.style.use('default')
plt.style.use('dark_background')
sns.pairplot(train_raw[['Sex','Age','RestingBP','Cholesterol','HeartDisease']],
             hue='HeartDisease',
             palette='colorblind')
plt.savefig('plots/pairplot_modelled.png',format='png',dpi=300)

''' --------------------------Prepare data-------------------'''

model_vars = ['Age','RestingBP','Cholesterol']
sex_conv = (train_raw['Sex']=='M').astype(int)

train2 = train_raw[model_vars]
train2 = train2.dropna()
train2.loc[:,'Sex'] = sex_conv.copy()
train2 = train2.values

scaler = StandardScaler().fit(train2)
train_std = scaler.transform(train2)

X_train = train_std
Y_train = train_raw['HeartDisease'].values

''' --------------------------Classify--------------------'''

model = LogisticRegressionCV(Cs=10,cv=5,scoring='f1_macro')
model.fit(X_train,Y_train)

preds_train = model.predict(X_train)
probs_train = model.predict_proba(X_train)

conf = confusion_matrix(Y_train,preds_train)
rep = classification_report(Y_train,preds_train)
f1 = f1_score(Y_train,preds_train,average='macro')
auc = roc_auc_score(Y_train,preds_train)

results = {'conf':conf,
           'rep':rep,
           'f1':f1,
           'auc':auc}

# Plot ROC curve
fpr,tpr,thresh = roc_curve(Y_train,probs_train[:,1],drop_intermediate=True)

# plt.style.use('default')
plt.style.use('dark_background')
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(fpr,tpr,color='purple',label=f'AUC={auc:0.2f}')
ax.plot([0,1],[0,1],ls='--',color='gray')
plt.title('Receiver Operating Characteristic')
plt.legend()
plt.ylabel('False Positive Rate')
plt.xlabel('True Positive Rate')
plt.savefig('plots/roc.png',format='png',dpi=300,bbox_inches='tight')

''' --------------------------Save model--------------------'''

pickle.dump(scaler,open('std_scaler.sav','wb'))
pickle.dump(model,open('cls_model.sav','wb'))
pickle.dump(results,open('output/model_results.sav','wb'))

''' --------------------------Test--------------------'''

test_record = np.array([[40,140,289,1]])
test_rec_std = scaler.transform(test_record)
preds_test_rec = model.predict(test_rec_std)
probs_test_rec = model.predict_proba(test_rec_std)