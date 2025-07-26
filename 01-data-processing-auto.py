import pandas as pd
import numpy as np
import duckdb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import os

"""
Data Processing
Step 1: Split data into NonAuto and Auto (NonAuto doesn't need bi_limit_group, newest_veh_age, telematics_ind)
Step 2: Split train into train and validation
Step 3: Use KNN to impute missing categorical values for these categorical variables: acq_method (missing), pol_edeliv_ind (-2 as missing), telematics_ind (-1 as missing)
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

# Step 1: Split data into NonAuto and Auto (NonAuto doesn't need bi_limit_group, newest_veh_age, telematics_ind)
def auto(data):
    conn = duckdb.connect()
    conn.register("data", data)
    auto_query = """
        SELECT * FROM data 
        WHERE bi_limit_group != 'NonAuto' AND newest_veh_age != -20 AND telematics_ind != -2;
    """
    df = conn.execute(auto_query).fetch_df()
    conn.close()
    return df

auto_train_df = auto(train)
auto_test_df = auto(test)

# Step 2: Split train into train and validation (Auto)
X = auto_train_df.drop(["call_counts" , "id"], axis=1)
y = auto_train_df["call_counts"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Step 3: Use KNN to impute missing categorical values for these categorical variables: acq_method (missing), pol_edeliv_ind (-2 and -1 as missing), telematics_ind (-1 as missing)

# Prepare data for imputing acq_method using KNN
def knn_prep(X):
    data = X.copy()
    data["index"] = X.index

    conn = duckdb.connect()
    conn.register("data", data)
    
    query = """
    CREATE OR REPLACE TABLE knn_prep AS
        SELECT 
            index, -- keeping track of index
            "12m_call_history",
            ann_prm_amt,
            home_lot_sq_footage,
            household_policy_counts,
            newest_veh_age,
            tenure_at_snapshot,
            trm_len_mo,
            CAST(CASE 
                WHEN acq_method = 'method1' THEN 1
                WHEN acq_method = 'method2' THEN 2
                WHEN acq_method = 'method3' THEN 3
                WHEN acq_method = 'method4' THEN 4
                ELSE NULL 
            END AS INTEGER) AS acq_method_encoded
        FROM data;
    """
    conn.execute(query)
    train = """
        SELECT * FROM knn_prep
        WHERE acq_method_encoded IS NOT NULL;
    """
    test = """
        SELECT * FROM knn_prep
        WHERE acq_method_encoded IS NULL;
    """

    train = conn.execute(train).fetch_df()
    test = conn.execute(test).fetch_df()

    X_train = train.drop(["acq_method_encoded", "index"], axis=1)
    y_train = train[["index","acq_method_encoded"]] # has values for the target

    X_test = test.drop(["acq_method_encoded", "index"], axis=1)
    y_test = test[["index","acq_method_encoded"]] # need to be imputed

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    conn.close()

    return X_train, y_train, X_test, y_test


def impute_df(X, acq_method_imputed):
    data = X.copy()
    data["index"] = data.index

    conn = duckdb.connect()
    conn.register("data", data)
    conn.register("acq_method_imputed", acq_method_imputed)

    query = """
    WITH cte AS (SELECT 
        a.*,
        CASE 
            WHEN a.acq_method = 'missing' THEN 
                CASE 
                    WHEN b.acq_method_encoded = 1 THEN 'method1'
                    WHEN b.acq_method_encoded = 2 THEN 'method2'
                    WHEN b.acq_method_encoded = 3 THEN 'method3'
                    WHEN b.acq_method_encoded = 4 THEN 'method4'
                    ELSE NULL
                END
            ELSE a.acq_method
        END AS acq_method_filled
    FROM data AS a
    LEFT JOIN acq_method_imputed AS b
    ON a.index = b.index)

    SELECT * EXCLUDE (acq_method)
    FROM cte;
    """
    X_final = conn.execute(query).fetch_df()
    X_final.set_index("index", inplace=True)
    conn.close()
    return X_final



# Impute X_train acq_method
X_train_knn, y_train_knn, X_test_knn, y_test_knn = knn_prep(X_train)

# Train KNNClassifier imputer
knn_imputer = KNeighborsClassifier(n_neighbors=5)  # you can tune this
knn_imputer.fit(X_train_knn, y_train_knn["acq_method_encoded"].values)

y_pred = knn_imputer.predict(X_test_knn)
y_test_knn["acq_method_encoded"] = y_pred
acq_method_imputed = pd.concat([y_train_knn, y_test_knn], axis=0)

X_train = impute_df(X_train, acq_method_imputed)



# Impute X_val acq_method
_, y_train_knn, X_test_knn, y_test_knn = knn_prep(X_val)
y_pred = knn_imputer.predict(X_test_knn)
y_test_knn["acq_method_encoded"] = y_pred
acq_method_imputed = pd.concat([y_train_knn, y_test_knn], axis=0)

X_val = impute_df(X_val, acq_method_imputed)

# Impute test acq_method
_, y_train_knn, X_test_knn, y_test_knn = knn_prep(auto_test_df)
y_pred = knn_imputer.predict(X_test_knn)
y_test_knn["acq_method_encoded"] = y_pred
acq_method_imputed = pd.concat([y_train_knn, y_test_knn], axis=0)

auto_test_df = impute_df(auto_test_df, acq_method_imputed)



# Prepare data for imputing pol_edeliv_ind (-2 and missing) using KNN
def knn_prep(X):
    data = X.copy()
    data["index"] = X.index

    conn = duckdb.connect()
    conn.register("data", data)
    
    query = """
    CREATE OR REPLACE TABLE knn_prep AS
        SELECT 
            index, -- keeping track of index
            "12m_call_history",
            ann_prm_amt,
            home_lot_sq_footage,
            household_policy_counts,
            newest_veh_age,
            tenure_at_snapshot,
            trm_len_mo,
            CAST(CASE 
                WHEN pol_edeliv_ind = 0 THEN 0
                WHEN pol_edeliv_ind = 1 THEN 1
                WHEN pol_edeliv_ind = -1 THEN -1
                ELSE NULL 
            END AS INTEGER) AS pol_edeliv_ind_encoded
        FROM data;
    """
    conn.execute(query)
    train = """
        SELECT * FROM knn_prep
        WHERE pol_edeliv_ind_encoded IS NOT NULL;
    """
    test = """
        SELECT * FROM knn_prep
        WHERE pol_edeliv_ind_encoded IS NULL;
    """

    train = conn.execute(train).fetch_df()
    test = conn.execute(test).fetch_df()

    X_train = train.drop(["pol_edeliv_ind_encoded", "index"], axis=1)
    y_train = train[["index","pol_edeliv_ind_encoded"]] # has values for the target

    X_test = test.drop(["pol_edeliv_ind_encoded", "index"], axis=1)
    y_test = test[["index","pol_edeliv_ind_encoded"]] # need to be imputed

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    conn.close()

    return X_train, y_train, X_test, y_test


def impute_df(X, pol_edeliv_ind_imputed):
    data = X.copy()
    data["index"] = data.index

    conn = duckdb.connect()
    conn.register("data", data)
    conn.register("pol_edeliv_ind_imputed", pol_edeliv_ind_imputed)

    query = """
    WITH cte AS (SELECT 
        a.*,
        CASE 
            WHEN a.pol_edeliv_ind = -2 THEN b.pol_edeliv_ind_encoded
            ELSE a.pol_edeliv_ind
        END AS pol_edeliv_ind_filled
    FROM data AS a
    LEFT JOIN pol_edeliv_ind_imputed AS b
    ON a.index = b.index)

    SELECT * EXCLUDE (pol_edeliv_ind)
    FROM cte;
    """
    X_final = conn.execute(query).fetch_df()
    X_final.set_index("index", inplace=True)
    conn.close()
    return X_final



# Impute X_train pol_edeliv_ind
X_train_knn, y_train_knn, X_test_knn, y_test_knn = knn_prep(X_train)

# Train KNNClassifier imputer
knn_imputer = KNeighborsClassifier(n_neighbors=5)  # you can tune this
knn_imputer.fit(X_train_knn, y_train_knn["pol_edeliv_ind_encoded"].values)

y_pred = knn_imputer.predict(X_test_knn)
y_test_knn["pol_edeliv_ind_encoded"] = y_pred
pol_edeliv_ind_imputed = pd.concat([y_train_knn, y_test_knn], axis=0)

X_train = impute_df(X_train, pol_edeliv_ind_imputed)



# Impute X_val pol_edeliv_ind
_, y_train_knn, X_test_knn, y_test_knn = knn_prep(X_val)
y_pred = knn_imputer.predict(X_test_knn)
y_test_knn["pol_edeliv_ind_encoded"] = y_pred
pol_edeliv_ind_imputed = pd.concat([y_train_knn, y_test_knn], axis=0)

X_val = impute_df(X_val, pol_edeliv_ind_imputed)

# Impute test pol_edeliv_ind
_, y_train_knn, X_test_knn, y_test_knn = knn_prep(auto_test_df)
y_pred = knn_imputer.predict(X_test_knn)
y_test_knn["pol_edeliv_ind_encoded"] = y_pred
pol_edeliv_ind_imputed = pd.concat([y_train_knn, y_test_knn], axis=0)

auto_test_df = impute_df(auto_test_df, pol_edeliv_ind_imputed)



# Prepare data for imputing telematics_ind (-1 as missing) using KNN
def knn_prep(X):
    data = X.copy()
    data["index"] = X.index

    conn = duckdb.connect()
    conn.register("data", data)
    
    query = """
    CREATE OR REPLACE TABLE knn_prep AS
        SELECT 
            index, -- keeping track of index
            "12m_call_history",
            ann_prm_amt,
            home_lot_sq_footage,
            household_policy_counts,
            newest_veh_age,
            tenure_at_snapshot,
            trm_len_mo,
            CAST(CASE 
                WHEN telematics_ind = 0 THEN 0
                WHEN telematics_ind = 1 THEN 1
                ELSE NULL 
            END AS INTEGER) AS telematics_ind_encoded
        FROM data;
    """
    conn.execute(query)
    train = """
        SELECT * FROM knn_prep
        WHERE telematics_ind_encoded IS NOT NULL;
    """
    test = """
        SELECT * FROM knn_prep
        WHERE telematics_ind_encoded IS NULL;
    """

    train = conn.execute(train).fetch_df()
    test = conn.execute(test).fetch_df()

    X_train = train.drop(["telematics_ind_encoded", "index"], axis=1)
    y_train = train[["index","telematics_ind_encoded"]] # has values for the target

    X_test = test.drop(["telematics_ind_encoded", "index"], axis=1)
    y_test = test[["index","telematics_ind_encoded"]] # need to be imputed

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    conn.close()

    return X_train, y_train, X_test, y_test


def impute_df(X, telematics_ind_imputed):
    data = X.copy()
    data["index"] = data.index

    conn = duckdb.connect()
    conn.register("data", data)
    conn.register("telematics_ind_imputed", telematics_ind_imputed)

    query = """
    WITH cte AS (SELECT 
        a.*,
        CASE 
            WHEN a.telematics_ind = -1 THEN b.telematics_ind_encoded
            ELSE a.telematics_ind
        END AS telematics_ind_filled
    FROM data AS a
    LEFT JOIN telematics_ind_imputed AS b
    ON a.index = b.index)

    SELECT * EXCLUDE (telematics_ind)
    FROM cte;
    """
    X_final = conn.execute(query).fetch_df()
    X_final.set_index("index", inplace=True)
    conn.close()
    return X_final



# Impute X_train telematics_ind
X_train_knn, y_train_knn, X_test_knn, y_test_knn = knn_prep(X_train)

# Train KNNClassifier imputer
knn_imputer = KNeighborsClassifier(n_neighbors=5)  # you can tune this
knn_imputer.fit(X_train_knn, y_train_knn["telematics_ind_encoded"].values)

y_pred = knn_imputer.predict(X_test_knn)
y_test_knn["telematics_ind_encoded"] = y_pred
telematics_ind_imputed = pd.concat([y_train_knn, y_test_knn], axis=0)

X_train = impute_df(X_train, telematics_ind_imputed)



# Impute X_val telematics_ind
_, y_train_knn, X_test_knn, y_test_knn = knn_prep(X_val)
y_pred = knn_imputer.predict(X_test_knn)
y_test_knn["telematics_ind_encoded"] = y_pred
telematics_ind_imputed = pd.concat([y_train_knn, y_test_knn], axis=0)

X_val = impute_df(X_val, telematics_ind_imputed)

# Impute test telematics_ind
_, y_train_knn, X_test_knn, y_test_knn = knn_prep(auto_test_df)
y_pred = knn_imputer.predict(X_test_knn)
y_test_knn["telematics_ind_encoded"] = y_pred
telematics_ind_imputed = pd.concat([y_train_knn, y_test_knn], axis=0)

auto_test_df = impute_df(auto_test_df, telematics_ind_imputed)



if not os.path.exists("data/model_data_auto"):
    os.makedirs("data/model_data_auto")
X_train.to_csv("data/model_data_auto/X_train.csv", index=False)
X_val.to_csv("data/model_data_auto/X_val.csv", index=False)
auto_test_df.to_csv("data/model_data_auto/X_test.csv", index=False)


