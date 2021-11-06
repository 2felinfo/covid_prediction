# Importing the libraries
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import warnings
import pickle
import joblib

warnings.filterwarnings('ignore')

data=pd.read_csv("file.csv")#importation du DataSet 
data1=data #création d'une copie du Dataset
NoneFeatures=['batch_date','test_name','swab_type','rapid_flu_results','rapid_strep_results',
'cough_severity',                    
'sob_severity',                     
'cxr_findings',                     
'cxr_impression',                   
'cxr_label',                        
'cxr_link','days_since_symptom_onset']
for item in NoneFeatures:
    data1.drop(columns=item, inplace=True)
category = data1.select_dtypes(include=['object']).copy()
l_category = list(category)
for item in l_category[1:]:
    data1[item] = data1[item].fillna(data1[item].mode()[0])
MissedNumericalDataFeatures = ['temperature','pulse','sys','dia','rr','sats']
for item in MissedNumericalDataFeatures:
    data1[item] = data1[item].fillna(data1[item].median())
data2=data1
data1 = data1.drop(data1[data1.age < 12].index)
data1 = data1.drop(data1[data1.temperature <36].index)
data1 = data1.drop(columns=['diarrhea','diabetes','rhonchi'])

from sklearn import *
train_x=data1.drop(['covid19_test_results'],axis=1)
L = preprocessing.LabelEncoder ()
matchresults = L.fit_transform (list (data1['covid19_test_results']))
train_y=list(matchresults)
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(train_x, train_y, test_size= 0.4)  
from sklearn.preprocessing import StandardScaler      
from sklearn.linear_model import LogisticRegression  
classifier= LogisticRegression()  
classifier.fit(x_train, y_train)  
print(data1.columns)
# Saving model to disk
#pickle.dump(classifier, open('model.pkl','wb'))
joblib.dump(classifier, 'finalized_model.joblib')


'''
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
'''
