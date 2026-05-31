from sklearn.model_selection import train_test_split

from s1_data.a0_setup_directories import auto_model_dir
from s1_data.db_utils import load_df, save_df
from s1_data.transform_utils import apply_transformer, fit_transformer, save_transformer

import duckdb
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

base_folder = "data"
database = "TravelersPolicyHolderCall.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=False)

"""
Hurdle model data prep for Auto segment.

Step 1: Apply cap and log1p transformations and build hurdle data layout
    - Auto_{train,test}_binary: all rows; target = (call_counts > 0)
    - Auto_{train,test}_count:  train filtered to call_counts > 0; test unfiltered
    - test sets stay unfiltered so both stage models can score every test row

Step 2: Stage 1 - Logistic Regression data prep (binary classification)
    Split train/val, one-hot encode (drop_first=True), standardize numerics, save.

Step 3: Stage 2 - Negative Binomial data prep (count regression)
    Split train/val, one-hot encode (drop_first=True), standardize numerics, save.
"""

# Step 1: Apply cap and log1p transformations and build hurdle data layout
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
                , LEAST(household_policy_counts, 2) AS _household_policy_counts_le2
                , GREATEST(LEAST(household_policy_counts, 4) - 2, 0) AS _household_policy_counts_gt2
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
            , _household_policy_counts_le2 AS household_policy_counts_le2
            , _household_policy_counts_gt2 AS household_policy_counts_gt2
            , newest_veh_age AS newest_veh_age
            , pol_edeliv_ind_filled AS pol_edeliv_ind
            , prdct_sbtyp_grp AS prdct_sbtyp_grp
            , product_sbtyp AS product_sbtyp
            , telematics_ind AS telematics_ind
            , _tenure_at_snapshot AS tenure_at_snapshot
            {target_outer}
        FROM cte;

        """)


nominal_cat = [
    "acq_method", "bi_limit_group", "geo_group", "pol_edeliv_ind",
    "prdct_sbtyp_grp", "product_sbtyp", "telematics_ind",
]
numeric_cols = [
    "12m_call_history", "ann_prm_amt", "home_lot_sq_footage",
    "household_policy_counts_le2", "household_policy_counts_gt2",
    "newest_veh_age", "tenure_at_snapshot",
]

#################### Step 2: Stage 1 - Logistic Regression Data Prep ####################
train_binary = load_df(conn, "Auto_train_binary")
test_binary = load_df(conn, "Auto_test_binary")

X = train_binary.drop(["nonzero_call"], axis=1)
y = train_binary[["id", "nonzero_call"]]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_encoded, binary_transformer = fit_transformer(
    X_train,
    nominal_cat=nominal_cat,
    drop_first=True,
    numeric_cols=numeric_cols,
    standardize=True,
)
X_val_encoded = apply_transformer(X_val, binary_transformer)
test_encoded = apply_transformer(test_binary, binary_transformer)

save_transformer(binary_transformer, os.path.join(auto_model_dir, "auto_binary_transformer.pkl"))

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

print("Stage 1 Binary Classification Auto Dataset Prep is done.")

#################### Step 3: Stage 2 - Negative Binomial Data Prep ####################
train_count_nb = load_df(conn, "Auto_train_count")
test_count_nb = load_df(conn, "Auto_test_count")

X = train_count_nb.drop(["call_counts"], axis=1)
y = train_count_nb[["id", "call_counts"]]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_encoded, count_nb_transformer = fit_transformer(
    X_train,
    nominal_cat=nominal_cat,
    drop_first=True,
    numeric_cols=numeric_cols,
    standardize=True,
)
X_val_encoded = apply_transformer(X_val, count_nb_transformer)
test_encoded = apply_transformer(test_count_nb, count_nb_transformer)

save_transformer(count_nb_transformer, os.path.join(auto_model_dir, "auto_count_nb_transformer.pkl"))

tables = {
    "X_train_auto_count_nb": X_train_encoded,
    "X_val_auto_count_nb": X_val_encoded,
    "test_auto_count_nb": test_encoded,
    "y_train_auto_count_nb": y_train,
    "y_val_auto_count_nb": y_val
}

for table_name, df in tables.items():
    save_df(conn, df, table_name, add_id=False)

print(conn.execute("SHOW TABLES").fetchall())

print("Stage 2 Count Regression (NB) Auto Dataset Prep is done.")


conn.close()