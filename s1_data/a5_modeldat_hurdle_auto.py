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

# Step 1: Apply cap and log1p transformations
# Hurdle data layout:
#   - binary stage: all train rows; target = (call_counts > 0) as 0/1
#   - count  stage: train rows WHERE call_counts > 0; target = call_counts
#   - test  stays unfiltered for both stages (both models scored on full test)
for stage in ["binary", "count"]:
    for data in ["train", "test"]:
        if data == "train":
            if stage == "binary":
                target_inner = ", IF(call_counts > 0, 1, 0) AS nonzero_call"
                target_outer = ", nonzero_call AS nonzero_call"
                row_filter = ""
            else:  # count
                target_inner = ", call_counts"
                target_outer = ", call_counts AS call_counts"
                row_filter = "WHERE call_counts > 0"
        else:  # test (no target, no row filter)
            target_inner = ""
            target_outer = ""
            row_filter = ""

        conn.execute(f"""
        CREATE OR REPLACE TABLE Auto_{data}_{stage} AS
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
                , IF(household_group = '2autodwelling', 1, 0) AS _household_group
                , LEAST(household_policy_counts, 4) AS _household_policy_counts
                , newest_veh_age
                , pol_edeliv_ind_filled
                , prdct_sbtyp_grp
                , product_sbtyp
                , telematics_ind
                , LOG(1 + LEAST(tenure_at_snapshot, 500)) AS _tenure_at_snapshot
                {target_inner}
            FROM Auto_{data}_imputed
            {row_filter}
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
            , pol_edeliv_ind_filled AS pol_edeliv_ind
            , prdct_sbtyp_grp AS prdct_sbtyp_grp
            , product_sbtyp AS product_sbtyp
            , telematics_ind AS telematics_ind
            , _tenure_at_snapshot AS tenure_at_snapshot
            {target_outer}
        FROM cte;

        """)


train_binary = load_df(conn, "Auto_train_binary", delete_id=False)
test_binary = load_df(conn, "Auto_test_binary", delete_id=False)

# Step 2: Split training data into train and validation sets
X = train_binary.drop(["nonzero_call"], axis=1)
y = train_binary[["id", "nonzero_call"]]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# Step 3: One-hot encode nominal categorical variables
binary_cat = ["digital_contact_ind", "has_prior_carrier", "household_group"]
nominal_cat = ["acq_method", "bi_limit_group", "geo_group", "pol_edeliv_ind", 
                "prdct_sbtyp_grp", "product_sbtyp", "telematics_ind"]

X_train_encoded = pd.get_dummies(X_train, columns=nominal_cat, drop_first=True, dtype="int8")
X_val_encoded = pd.get_dummies(X_val, columns=nominal_cat, drop_first=True, dtype="int8")
test_encoded = pd.get_dummies(test_binary, columns=nominal_cat, drop_first=True, dtype="int8")

X_val_encoded = X_val_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
test_encoded = test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)

tables = {
    "X_train_auto_binary": X_train_encoded,
    "X_val_auto_binary": X_val_encoded,
    "test_auto_binary": test_encoded,
    "y_train_auto_binary": y_train,
    "y_val_auto_binary": y_val
}

for table_name, df in tables.items():
    save_df(conn, df, table_name, add_id=False)

print(conn.execute("SHOW TABLES").fetchall())
conn.close()