import pandas as pd
import numpy as np
import duckdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import os

"""
Data Processing
Step 1: split data into NonAuto and Auto (NonAuto doesn't need bi_limit_group, newest_veh_age, telematics_ind)
Step 2: split train into train and validation
Step 3: Use KNN to impute missing categorical values for these categorical variables: acq_method (missing), pol_edeliv_ind (-2 and -1 as missing), telematics_ind (-1 as missing)
- Training data: NonAuto Train, Auto Train
- Validation data: NonAuto Val, Auto Val
- Test: NonAuto Test, Auto Test
** Use NonAuto Train to train KNN to impute NonAuto Train, NonAuto Val, and NonAuto Test
** Use Auto Train to train KNN to impute Auto Train, Auto Val, and Auto Test
Step 4: Encoding categorical (one-hot for nominal and ordinal mapping for ordinal categorical)
** Remember to reindex the validation and test sets to match the one-hot encoded columns of the training set. X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
Step 5: Scale Numerical (normalization or standardization) for train and use the same mean and st dev for validation and test sets

"""
train = pd.read_csv("data/train_data.csv")
test = pd.read_csv("data/test_data.csv")

random_state = 42

def non_auto(data):
    conn = duckdb.connect()
    conn.register("data", data)
    nonauto_query = """
        WITH cte AS (
            SELECT * FROM data 
            WHERE bi_limit_group = 'NonAuto' AND newest_veh_age = -20 AND telematics_ind = -2
        )
        SELECT * EXCLUDE (bi_limit_group, newest_veh_age, telematics_ind)
        FROM cte;
    """
    df = conn.execute(nonauto_query).fetch_df()
    conn.close()
    return df 

nonauto_train_df = non_auto(train)
nonauto_test_df = non_auto(test)