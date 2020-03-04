#Sepsis Detection Using Physiological Data
#Non-Temporal Approach


#----------------Packages------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Get Data File Names
path =r'C:\Users\91972\Desktop\Sepsis\training'
filenames = glob.glob(path + "/*.psv")


#-----------------Temporary Data Frame-----------------------------------------
df= []


#-----------------Concating All PSV File Data into Dataset Variable------------
for filename in filenames:
    df.append(pd.read_csv(filename,sep='|'))
Dataset = pd.concat(df,axis=0,ignore_index=True)

#-----------------Visualize Missing Values In Dataset--------------------------
missing_values = Dataset.isnull().sum() / len(Dataset)


#-----------------Feature Engineering------------------------------------------
feature_engineered_attributes= missing_values[missing_values <= 0.8]
feature_engineered_attributes_arr=[]
feature_engineered_attributes_arr=feature_engineered_attributes.index
Drop_attributes=[]
for x in Dataset:
        if x not in feature_engineered_attributes_arr:
            Drop_attributes.append(x)
Dataset=Dataset.drop(columns=Drop_attributes)
#------------------------------------------------------------------------------


missing_values.sort_values(inplace=True)
missing_values

missing_values = missing_values.to_frame()
missing_values.columns = ['Count']
missing_values.index.names = ['Name']
missing_values['Name'] = missing_values.index


sns.set(style="whitegrid", color_codes=True)
sns.barplot(x = 'Name', y = 'Count', data=missing_values)
plt.xticks(rotation = 90)
plt.show()


#-----------------Visualize Missing Values In Dataset After Feature Engineering
missing_values = Dataset.isnull().sum() / len(Dataset)
missing_values.sort_values(inplace=True)
missing_values

missing_values = missing_values.to_frame()
missing_values.columns = ['Count']
missing_values.index.names = ['Name']
missing_values['Name'] = missing_values.index


sns.set(style="whitegrid", color_codes=True)
sns.barplot(x = 'Name', y = 'Count', data=missing_values)
plt.xticks(rotation = 90)
plt.show()


#-----------------One Hot Encoding---------------------------------------------


#Heart rate for a healthy adult is between 60 and 100. For a healthy infant it is between 70 and 190. Creating a new feature custom_hr , which is categorical variable having three values Normal, Abnormal and Missing
def feature_engineer_hr(Dataset):
    Dataset.loc[(Dataset['HR'] >= 100) & (Dataset['Age'] >= 10 ),
            'custom_HR'] = 'Abnormal'
    Dataset.loc[(Dataset['HR'] < 100) & (Dataset['HR'] > 60) & (Dataset['Age'] >= 10 ),
            'custom_HR'] = 'Normal'
    Dataset.loc[(Dataset['HR'] >= 70) & (Dataset['HR'] < 190) & (Dataset['Age'] < 10 ),
            'custom_HR'] = 'Normal'
    Dataset.loc[((Dataset['HR'] < 70) | (Dataset['HR'] >= 190)) & (Dataset['Age'] < 10 ),
            'custom_HR'] = 'Abnormal'
    Dataset['custom_HR'].fillna('Missing', inplace=True)
    return Dataset

Dataset=feature_engineer_hr(Dataset)
Dataset=Dataset.drop(columns='HR')

#Temperature for a healthy human being is between 36.4 degree C to 37.6 degree C. Creating a new feature custom_temp , which is categorical variable having three values Normal, Abnormal and Missing
def feature_engineer_temp(Dataset):
    Dataset.loc[(Dataset['Temp'] >= 36.4) & (Dataset['Temp'] < 37.6), 
            'custom_TEMP'] = 'Normal'
    Dataset.loc[(Dataset['Temp'] < 36.4) | (Dataset['Temp'] >= 37.6), 
            'custom_TEMP'] = 'Abnormal'

    Dataset['custom_TEMP'].fillna('Missing', inplace=True)
    return Dataset

Dataset=feature_engineer_temp(Dataset)
Dataset=Dataset.drop(columns='Temp')


#Age-Categorizing patient based on age to old, infant and Child/adult
def feature_engineer_age(Dataset):
    Dataset.loc[Dataset['Age'] >=65, 'custom_AGE'] = 'Old'
    Dataset.loc[Dataset['Age'] <1, 'custom_AGE'] = 'Infant'
    Dataset.loc[(Dataset['Age'] >=1) & (Dataset['Age'] <65), 
            'custom_AGE'] = 'Child/Adult'
    return Dataset
Dataset=feature_engineer_age(Dataset)


#O2Stat for a healthy adult is between 90 and 100 for healthy human beings. Create a new categorical variable custom_o2stat with levels normal, abnormal and missing
def feature_engineer_o2stat(Dataset):
    Dataset.loc[(Dataset['O2Sat'] >= 90) & (Dataset['O2Sat'] < 100), 
            'custom_O2STAT'] = 'Normal'
    Dataset.loc[(Dataset['O2Sat'] < 90) & (Dataset['O2Sat'] >= 0), 
            'custom_O2STAT'] = 'Abnormal'

    Dataset['custom_O2STAT'].fillna('Missing', inplace=True)
    return Dataset
Dataset=feature_engineer_o2stat(Dataset)
Dataset=Dataset.drop(columns='O2Sat')


#SBP stands for Systolic blood pressure, It is the upper number while measuring Blood pressure. DBP stands for Diastolic blood pressure , It is the lower number while measuring Blood pressure. Using both these columns to categorize blood pressure as low, normal, elevated , high and missing
def feature_engineer_blood_pressure(Dataset):
    Dataset.loc[(Dataset['SBP'] <90) & (Dataset['DBP'] <60), 'custom_BP'] = 'low'

    Dataset.loc[(Dataset['SBP'].between(90,120, inclusive=True)) & 
            (Dataset['DBP'].between(60,80, inclusive=True)), 
            'custom_BP'] = 'Normal'


    Dataset.loc[(Dataset['SBP'].between(120,140, inclusive=True)) & 
            (Dataset['DBP'].between(80,90, inclusive=True)), 
            'custom_BP'] = 'Elevated'


    Dataset.loc[(Dataset['SBP'] > 140 ) & 
            (Dataset['DBP'] > 90 ), 'custom_BP'] = 'High'

    Dataset['custom_BP'].fillna('Missing', inplace=True)
    return Dataset
Dataset=feature_engineer_blood_pressure(Dataset)
Dataset=Dataset.drop(columns='DBP')
Dataset=Dataset.drop(columns='SBP')


#Respiration rate for healthy adults is between 12 and 20. Categorizing respiratory rate as normal and abnormal based on thresholds.
def feature_engineer_resp_rate(Dataset):
    Dataset.loc[(Dataset['Resp'].between(30,60)) & (Dataset['Age'] <1), 
            'custom_RESP'] = 'Normal'
    Dataset.loc[((Dataset['Resp'] < 30) | (Dataset['Resp'] > 60)) & 
             (Dataset['Age'] <1) ,'custom_RESP'] = 'Abnormal'


    Dataset.loc[(Dataset['Resp'].between(24,40)) & (Dataset['Age'].between(1,3)), 
            'custom_RESP'] = 'Normal'
    Dataset.loc[((Dataset['Resp'] < 24) | (Dataset['Resp'] > 40)) & 
             (Dataset['Age'].between(1,3)) ,'custom_RESP'] = 'Abnormal'


    Dataset.loc[(Dataset['Resp'].between(22,34)) & (Dataset['Age'].between(3,6)), 
            'custom_RESP'] = 'Normal'
    Dataset.loc[((Dataset['Resp'] < 22) | (Dataset['Resp'] > 34)) & 
             (Dataset['Age'].between(3,6)) ,'custom_RESP'] = 'Abnormal'


    Dataset.loc[(Dataset['Resp'].between(18,30)) & (Dataset['Age'].between(6,12)), 
            'custom_RESP'] = 'Normal'
    Dataset.loc[((Dataset['Resp'] < 18) | (Dataset['Resp'] > 30)) & 
             (Dataset['Age'].between(6,12)) ,'custom_RESP'] = 'Abnormal'


    Dataset.loc[(Dataset['Resp'].between(12,20)) & (Dataset['Age'] >12), 
            'custom_RESP'] = 'Normal'
    Dataset.loc[((Dataset['Resp'] < 12) | (Dataset['Resp'] > 20)) & (Dataset['Age'] >12),
            'custom_RESP'] = 'Abnormal'

    Dataset['custom_RESP'].fillna('Missing', inplace=True)

    return Dataset
Dataset=feature_engineer_resp_rate(Dataset)
Dataset=Dataset.drop(columns='Resp')
Dataset=Dataset.drop(columns='Age') #We Are Going To Drop Attribute Age Here Because In Respiration Rate , Age Is Required.



#-----------------Dependent & Undependent Variable-----------------------------   
X = Dataset.iloc[:,:-1].values
y = Dataset.iloc[:, -1].values


#----------------One Hot Encoding----------------------------------------------

    
    


