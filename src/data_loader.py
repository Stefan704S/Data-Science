# Importing data

import pandas as pd


def data_load(path="data/raw/data.slsx"):
    data = pd.read_excel(path)
    return data


# ------------------------------------------------------------------------------------------------------------------------------------
# Function to Standardization variables

from sklearn.preprocessing import StandardScaler


def standardize(x: pd.DataFrame):
    # Standardization of variables to test the first hypothesis. In this part of the code, we select the data we want to standardize, create a new matrix, and then add the standardized data from the initial database to it.

    scaler = StandardScaler()
    return scaler.fit_transform(x)