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
Step 1: Apply cap and log1p transformations (drop pay_type_code: identical to product_sbtyp)
Step 2: Split training data into train and validation sets
Step 3: One-hot encode nominal categorical variables
Step 4: Standardize numeric variables and save to DuckDB
"""

# Step 1: Apply cap and log1p transformations
for data in ["train", "test"]:
    target_col = ', call_counts AS call_counts' if data == "train" else ""
    conn.execute(f"""
    CREATE OR REPLACE TABLE NonAuto_{data}_baseline AS
    WITH cte AS (
        SELECT
            id
            , LOG(1 + LEAST("12m_call_history", 30)) AS _12m_call_history
            , acq_method
            , LOG(1 + LEAST(ann_prm_amt, 7200)) AS _ann_prm_amt
            , IF(channel = 'Retail', 1, 0) AS _channel
            , digital_contact_ind
            , geo_group
            , has_prior_carrier
            , home_lot_sq_footage
            , IF(household_group = '1dwelling', 1, 0) AS _household_group
            , LEAST(household_policy_counts, 5) AS _household_policy_counts
            , pol_edeliv_ind_filled
            , prdct_sbtyp_grp
            , product_sbtyp
            , LOG(1 + LEAST(tenure_at_snapshot, 500)) AS _tenure_at_snapshot
            , IF(trm_len_mo = 12, 1, 0) AS _trm_len_mo
            {target_col}
        FROM NonAuto_{data}_imputed
    )
    SELECT 
        id AS id
        , _12m_call_history AS "12m_call_history"
        , acq_method AS acq_method
        , _ann_prm_amt AS ann_prm_amt
        , _channel AS channel
        , digital_contact_ind AS digital_contact_ind
        , geo_group AS geo_group
        , has_prior_carrier AS has_prior_carrier
        , home_lot_sq_footage AS home_lot_sq_footage
        , _household_group AS household_group
        , _household_policy_counts AS household_policy_counts
        , pol_edeliv_ind_filled AS pol_edeliv_ind
        , prdct_sbtyp_grp AS prdct_sbtyp_grp
        , product_sbtyp AS product_sbtyp
        , _tenure_at_snapshot AS tenure_at_snapshot
        , _trm_len_mo AS trm_len_mo
        {target_col}
    FROM cte;
    """)

train = load_df(conn, "NonAuto_train_baseline")
test = load_df(conn, "NonAuto_test_baseline")

# Step 2: Split training data into train and validation sets
X = train.drop(["call_counts"], axis=1)
y = train[["id", "call_counts"]]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: One-hot encode nominal categorical variables
binary_cat = ["channel", "digital_contact_ind", "has_prior_carrier", 
              "household_group", "trm_len_mo"]
nominal_cat = ["acq_method", "geo_group", "pol_edeliv_ind", 
                "prdct_sbtyp_grp", "product_sbtyp"]

X_train_encoded = pd.get_dummies(X_train, columns=nominal_cat, drop_first=True, dtype="int8")
X_val_encoded = pd.get_dummies(X_val, columns=nominal_cat, drop_first=True, dtype="int8")
test_encoded = pd.get_dummies(test, columns=nominal_cat, drop_first=True, dtype="int8")

X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
test_encoded = test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

# Step 4: Standardize numeric variables and save to DuckDB
numeric_cols = ["12m_call_history", "ann_prm_amt", "home_lot_sq_footage",
                "household_policy_counts", "tenure_at_snapshot"]

scaler = StandardScaler()
X_train_encoded[numeric_cols] = scaler.fit_transform(X_train_encoded[numeric_cols])
X_val_encoded[numeric_cols] = scaler.transform(X_val_encoded[numeric_cols])
test_encoded[numeric_cols] = scaler.transform(test_encoded[numeric_cols])

tables = {
    "X_train_nonauto_base": X_train_encoded,
    "X_val_nonauto_base": X_val_encoded,
    "test_nonauto_base": test_encoded,
    "y_train_nonauto": y_train,
    "y_val_nonauto": y_val
}

for table_name, df in tables.items():
    save_df(conn, df, table_name, add_id=False)

print(conn.execute("SHOW TABLES").fetchall())
conn.close()
