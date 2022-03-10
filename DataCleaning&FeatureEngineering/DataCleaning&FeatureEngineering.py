import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# import csv, skip lines with error
data = pd.read_csv("finantier_data_technical_test_dataset.csv", error_bad_lines=False, header=None)

# drop empty rows
data.dropna(how='all', inplace=True)

# make first row header
new_header = data.iloc[0] #grab the first row for the header
data = data[1:] #take the data less the header row
data.columns = new_header 

# change relevant columns to categorical and int data types
data = data.astype({
    'gender':'category',
    'SeniorCitizen':'category',
    'Partner':'category',
    'Dependents':'category',
    'tenure':'int64',
    'PhoneService':'category',
    'MultipleLines':'category',
    'InternetService':'category',
    'OnlineSecurity':'category',
    'OnlineBackup':'category',
    'DeviceProtection':'category',
    'TechSupport':'category',
    'StreamingTV':'category',
    'StreamingMovies':'category',
    'Contract':'category',
    'PaperlessBilling':'category',
    'PaymentMethod':'category',
    'Default':'category'
})

# change charges columns to float, missing values in TotalCharges changed to 0.0
data.MonthlyCharges = data.MonthlyCharges.astype(float)
data.TotalCharges = data.TotalCharges.str.replace(' ', '0.0').astype(float)

# square root transformation to rescale skewed TotalCharges column
data['sqrt_TotalCharges'] = np.sqrt(data['TotalCharges'])

# Normalise continuous variables
data['norm_MonthlyCharges'] = (data['MonthlyCharges']-data['MonthlyCharges'].min())/(data['MonthlyCharges'].max()-data['MonthlyCharges'].min())
data['norm_tenure'] = (data['tenure']-data['tenure'].min())/(data['tenure'].max()-data['tenure'].min())
data['normsqrt_TotalCharges'] = (data['sqrt_TotalCharges']-data['sqrt_TotalCharges'].min())/(data['sqrt_TotalCharges'].max()-data['sqrt_TotalCharges'].min())

# Bin continuous variables
quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']
quantile_list = [0, .25, .5, .75, 1.]
data['bin_TotalCharges'] = pd.qcut(
                                data['TotalCharges'], 
                                q=quantile_list,       
                                labels=quantile_labels)
data['bin_MonthlyCharges'] = pd.qcut(
                                data['MonthlyCharges'], 
                                q=quantile_list,       
                                labels=quantile_labels)
data['bin_tenure'] = pd.qcut(
                            data['tenure'], 
                            q=quantile_list,       
                            labels=quantile_labels)

# Perform integer coding 
inputdata = data.drop(columns=['customerID','tenure','MonthlyCharges','TotalCharges','log_TotalCharges','sqrt_TotalCharges'])
inputdata.gender.replace(('Female', 'Male'), (1, 0), inplace=True)
inputdata.Partner.replace(('Yes', 'No'), (1, 0), inplace=True)
inputdata.Dependents.replace(('Yes', 'No'), (1, 0), inplace=True)
inputdata.PhoneService.replace(('Yes', 'No'), (1, 0), inplace=True)
inputdata.MultipleLines.replace(('Yes', 'No', 'No phone service'), (2, 1,0), inplace=True)
inputdata.InternetService.replace(('DSL', 'Fiber optic', 'No'), (2, 1,0), inplace=True)
inputdata.OnlineSecurity.replace(('Yes', 'No','No internet service'), (2, 1,0), inplace=True)
inputdata.OnlineBackup.replace(('Yes', 'No','No internet service'), (2, 1,0), inplace=True)
inputdata.DeviceProtection.replace(('Yes', 'No','No internet service'), (2, 1,0), inplace=True)
inputdata.TechSupport.replace(('Yes', 'No','No internet service'), (2, 1,0), inplace=True)
inputdata.StreamingTV.replace(('Yes', 'No','No internet service'), (2, 1,0), inplace=True)
inputdata.StreamingMovies.replace(('Yes', 'No','No internet service'), (2, 1,0), inplace=True)
inputdata.Contract.replace(('Two year', 'One year', 'Month-to-month'), (2, 1,0), inplace=True)
inputdata.PaperlessBilling.replace(('Yes', 'No'), (1, 0), inplace=True)
inputdata.PaymentMethod.replace(('Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'), (3,2, 1,0), inplace=True)
inputdata.Default.replace(('Yes', 'No'), (1, 0), inplace=True)
inputdata.bin_TotalCharges.replace(('0-25Q','25-50Q','50-75Q','75-100Q'),(0,1,2,3),inplace=True)
inputdata.bin_MonthlyCharges.replace(('0-25Q','25-50Q','50-75Q','75-100Q'),(0,1,2,3),inplace=True)
inputdata.bin_tenure.replace(('0-25Q','25-50Q','50-75Q','75-100Q'),(0,1,2,3),inplace=True)

# Perform train test split
SEED = 2
data_train, data_test = train_test_split(
    inputdata, random_state=SEED, test_size=0.2, stratify=inputdata.Default
)
data_val, data_test = train_test_split(
    data_test, random_state=SEED, test_size=0.5, stratify=data_test.Default
)
data_train.to_pickle("../app/data_train.p")
data_val.to_pickle("../app/data_val.p")
data_test.to_pickle("../app/data_test.p")
