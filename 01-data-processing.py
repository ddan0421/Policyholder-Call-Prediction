import pandas as pd
import numpy as np
import duckdb
from sklearn.model_selection import train_test_split
import os

train = pd.read_csv("data/train_data.csv")
test = pd.read_csv("data/test_data.csv")

random_state = 42

# Step 1: Split data into NonAuto and Auto (NonAuto doesn't need bi_limit_group, newest_veh_age, telematics_ind)
conn = duckdb.connect()

nonauto_query = """
    WITH cte AS (
        SELECT * FROM train 
        WHERE bi_limit_group = 'NonAuto' AND newest_veh_age = -20 AND telematics_ind = -2
    )
    SELECT * EXCLUDE (bi_limit_group, newest_veh_age, telematics_ind)
    FROM cte;
"""

nonauto_df = conn.execute(nonauto_query).fetch_df()


auto_query = """
    SELECT * FROM train 
    WHERE bi_limit_group != 'NonAuto' AND newest_veh_age != -20 AND telematics_ind != -2;
"""

auto_df = conn.execute(auto_query).fetch_df()

conn.close()

# Step 2: Split train into train and validation
X = train.drop("call_counts", axis=1)
y = train["call_counts"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)





