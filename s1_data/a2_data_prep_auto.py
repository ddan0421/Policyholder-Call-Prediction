import pandas as pd
from sklearn.model_selection import train_test_split
from s1_data.db_utils import load_df, save_df
from s1_data.imputation_utils import apply_imputation, fit_and_impute_train

import duckdb
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

base_folder = "data"
database = "TravelersPolicyHolderCall.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=False)

"""
Step 1: Split data into NonAuto and Auto
(NonAuto doesn't need bi_limit_group, newest_veh_age, telematics_ind)
(Auto doesn't need trm_len_mo since it is always 6 and doesn't need channel since it is always retail)

Step 2: Hold out 20% of auto train for validation while fitting KNN imputation on the other 80%

Step 3: Use KNN to impute pol_edeliv_ind (-2 as missing)
- This project uses KNeighborsClassifier for imputation; consider KNNImputer as an alternative for directly imputing missing values.

Step 4: Export Auto_train_imputed and Auto_test_imputed to DuckDB
"""

# Step 1: Create Auto only data
for source in ["train", "test"]:
    auto_query = f"""
    CREATE OR REPLACE TABLE Auto_{source} AS
        WITH cte AS (
            SELECT * FROM {source}
            WHERE bi_limit_group != 'NonAuto' AND telematics_ind != -2
        )
        SELECT * EXCLUDE (trm_len_mo, channel) FROM cte;
    """
    conn.execute(auto_query)

# Step 2: Hold out validation split (full rows, including call_counts)
train = load_df(conn, "Auto_train", delete_id=False)
test = load_df(conn, "Auto_test", delete_id=False)

train_split, val_split = train_test_split(train, test_size=0.2, random_state=42)

# Step 3: Fit KNN imputer on train_split, then apply to val_split and test
train_split, knn_imputer, pol_scaler = fit_and_impute_train(conn, train_split)
val_split = apply_imputation(conn, val_split, knn_imputer, pol_scaler)
test      = apply_imputation(conn, test,      knn_imputer, pol_scaler)

# Step 4: Recombine train splits and export
train_imputed = pd.concat([train_split, val_split], axis=0)

save_df(conn, train_imputed, "Auto_train_imputed", add_id=False)
save_df(conn, test, "Auto_test_imputed", add_id=False)

print(conn.execute("SHOW TABLES").fetchall())
conn.close()
