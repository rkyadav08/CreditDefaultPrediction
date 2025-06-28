#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix


# In[2]:


def ModelPreprocessing(df):
    
#outliers Treatment
    below_70=df[df['person_age']<70]
    below_70.reset_index(drop=True,inplace=True)
    below_52=below_70[below_70['person_emp_length']<52]
    below_52.reset_index(drop=True,inplace=True)
    cr_data=below_52.copy()
    
#Missing Data Treatment
    cr_data.fillna({'loan_int_rate':cr_data['loan_int_rate'].mean()},inplace=True)
    cr_data_cat=cr_data.copy()
    
#Categorical Variable Treatment
    person_home_ownership=pd.get_dummies(cr_data_cat['person_home_ownership'],drop_first=True).astype(int)
    loan_intent=pd.get_dummies(cr_data_cat['loan_intent'],drop_first=True).astype(int)
    loan_grade=pd.get_dummies(cr_data_cat['loan_grade'],drop_first=True).astype(int)
    cr_data_cat['cb_person_default_on_file_bin']=np.where(cr_data_cat['cb_person_default_on_file']=='Y',1,0)
    data_to_scale = cr_data_cat.drop(['person_home_ownership', 'loan_intent', 'loan_grade', 'loan_status', 'cb_person_default_on_file', 'cb_person_default_on_file_bin'],axis=1)
    
#Scaling the Data
    scaler=StandardScaler()
    scaled_data=scaler.fit_transform(data_to_scale)
    scaled_df=pd.DataFrame(scaled_data,columns=['person_age', 'person_income', 'person_emp_length', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length'])
    scaled_combined=pd.concat([scaled_df,person_home_ownership,loan_intent,loan_grade,cr_data_cat['cb_person_default_on_file_bin'],cr_data_cat['loan_status']],axis=1)
    scaled_combined['cb_person_default_on_file']=scaled_combined['cb_person_default_on_file_bin']
    scaled_combined=scaled_combined.drop('cb_person_default_on_file_bin',axis=1)
    
#Features and target Creation
    target = scaled_combined['loan_status']
    features=scaled_combined.drop('loan_status',axis=1)
    
#Smote Balancing
    smote=SMOTE()
    balanced_features,balanced_target=smote.fit_resample(features,target)

#Return the final Dataset
    return data_to_scale,features,target,balanced_features,balanced_target
        




