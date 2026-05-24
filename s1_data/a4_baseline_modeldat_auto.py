import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from s1_data.db_utils import load_df, save_df

import duckdb
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

base_folder = "data"
database = "TravelersPolicyHolderCall.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=False)

"""
Step 1: Apply cap and log1p transformations
Step 2: Split training data into train and validation sets
Step 3: One-hot encode nominal categorical variables
Step 4: Standardize numeric variables and save to DuckDB
"""

# Step 1: Apply cap and log1p transformations
train = load_df(conn, "Auto_train_imputed", delete_id=False)
test = load_df(conn, "Auto_test_imputed", delete_id=False)

def log_cap_transform(conn, df):
    target_col = ', call_counts' if "call_counts" in df.columns else ""
    conn.register("input_df", df)
    data = conn.execute(f"""
    WITH cte AS (
        SELECT
            id
            , LOG(1 + LEAST("12m_call_history", 40)) AS _12m_call_history
            , acq_method
            , LOG(1 + LEAST(ann_prm_amt, 7200)) AS _ann_prm_amt
            , bi_limit_group
            , digital_contact_ind
            , geo_group
            , has_prior_carrier
            , home_lot_sq_footage
            , IF(household_group = '3autodwellingumb', 1, 0) AS _household_group
            , LEAST(household_policy_counts, 4) AS _household_policy_counts
            , newest_veh_age
            , pay_type_code
            , pol_edeliv_ind_filled
            , prdct_sbtyp_grp
            , product_sbtyp
            , telematics_ind
            , LOG(1 + LEAST(tenure_at_snapshot, 500)) AS _tenure_at_snapshot
            {target_col}
        FROM input_df
    )
    SELECT 
        id AS id
        , _12m_call_history AS "12m_call_history"
        , acq_method AS acq_method
        , _ann_prm_amt AS ann_prm_amt
        , bi_limit_group AS bi_limit_group
        , digital_contact_ind AS digital_contact_ind
        , geo_group AS geo_group
        , has_prior_carrier AS has_prior_carrier
        , home_lot_sq_footage AS home_lot_sq_footage
        , _household_group AS household_group
        , _household_policy_counts AS household_policy_counts
        , newest_veh_age AS newest_veh_age
        , pay_type_code AS pay_type_code
        , pol_edeliv_ind_filled AS pol_edeliv_ind
        , prdct_sbtyp_grp AS prdct_sbtyp_grp
        , product_sbtyp AS product_sbtyp
        , telematics_ind AS telematics_ind
        , _tenure_at_snapshot AS tenure_at_snapshot
        {target_col}
    FROM cte;

    """).fetch_df()
    conn.unregister("input_df")
    return data

train_transformed = log_cap_transform(conn, train)
test_transformed = log_cap_transform(conn, test)

# Step 2: Split training data into train and validation sets
X = train_transformed.drop(["call_counts"], axis=1)
y = train_transformed[["id", "call_counts"]]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: One-hot encode nominal categorical variables
binary_cat = ["digital_contact_ind", "has_prior_carrier", "household_group"]
nominal_cat = ["acq_method", "bi_limit_group", "geo_group", "pay_type_code",
               "pol_edeliv_ind", "prdct_sbtyp_grp", "product_sbtyp", "telematics_ind"]

X_train_encoded = pd.get_dummies(X_train, columns=nominal_cat, drop_first=True, dtype="int8")
X_val_encoded = pd.get_dummies(X_val, columns=nominal_cat, drop_first=True, dtype="int8")
test_encoded = pd.get_dummies(test_transformed, columns=nominal_cat, drop_first=True, dtype="int8")

X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
test_encoded = test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Step 4: Standardize numeric variables and save to DuckDB
numeric_cols = ["12m_call_history", "ann_prm_amt", "home_lot_sq_footage",
                "household_policy_counts", "newest_veh_age", "tenure_at_snapshot"]

scaler = StandardScaler()
X_train_encoded[numeric_cols] = scaler.fit_transform(X_train_encoded[numeric_cols])
X_val_encoded[numeric_cols] = scaler.transform(X_val_encoded[numeric_cols])
test_encoded[numeric_cols] = scaler.transform(test_encoded[numeric_cols])

tables = {
    "X_train_base": X_train_encoded,
    "X_val_base": X_val_encoded,
    "test_base": test_encoded,
    "y_train": y_train,
    "y_val": y_val
}

for table_name, df in tables.items():
    save_df(conn, df, table_name, add_id=False)

print(conn.execute("SHOW TABLES").fetchall())
conn.close()
