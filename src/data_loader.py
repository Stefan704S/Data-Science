# Data_loader.py
#-------------------------------------------------------------------------------------------------------------------------------------
# Importing librairies

import pandas as pd
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler


#-------------------------------------------------------------------------------------------------------------------------------------
# Importing the data

def data_load(path="data/raw/data.xlsx"):
    data = pd.read_excel(path)
    return data


#-------------------------------------------------------------------------------------------------------------------------------------
# Function for standardizing the data 
# In this part of the code, we select the data we want to standardize, create a new matrix, and then add the standardized data from the initial database to it.

def standardize(x: pd.DataFrame):
    scaler = StandardScaler()
    return scaler.fit_transform(x)
