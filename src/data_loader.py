# Importing data

import pandas as pd

def data_load(path="data/raw/data.slsx"):
    data = pd.read_excel(path)
    return(data)


# Function to Standardization variables

from sklearn.preprocessing import StandardScaler

def standardize(data : pd.DataFrame):

    # Standardization of variables to test the first hypothesis. In this part of the code, we select the data we want to standardize, create a new matrix, and then add the standardized data from the initial database to it.

    # Hypotesis 0
    x_full = data[["3Mth", "10Yd", "Inf", "Unmp", "CHF", "GDP", "SMI"]]
    scaler_full = StandardScaler()
    x_full_scaled = scaler_full.fit_transform(x_full)
    return x_full_scaled
