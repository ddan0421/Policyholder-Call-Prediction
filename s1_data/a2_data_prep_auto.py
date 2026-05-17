import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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
Step 1: Split data into NonAuto and Auto
(NonAuto doesn't need bi_limit_group, newest_veh_age, telematics_ind)
(Auto doesn't need trm_len_mo since it is always 6 and doesn't need channel since it is always retail)

Step 2: Hold out 20% of auto train for validation while fitting KNN imputation on the other 80%

Step 3: Use KNN to impute acq_method (missing)

Step 4: Export auto_train_imputed and auto_test_imputed to DuckDB
"""

# Step 1: Create Auto only data
for source in ["train", "test"]:
    auto_query = f"""
    CREATE OR REPLACE TABLE auto_{source} AS
        WITH cte AS (
            SELECT * FROM {source}
            WHERE bi_limit_group != 'NonAuto' AND telematics_ind != -2
        )
        SELECT * EXCLUDE (trm_len_mo, channel) FROM cte;
    """
    conn.execute(auto_query)

# Step 2: Hold out validation split (full rows, including call_counts)
train = load_df(conn, "auto_train", add_id=False)
test = load_df(conn, "auto_test", add_id=False)

train_split, val_split = train_test_split(train, test_size=0.2, random_state=42)

# Step 3: KNN impute acq_method (missing)
def knn_prep(conn, data, scaler=None):
    conn.register("data", data)

    query = """
    CREATE OR REPLACE TABLE knn_prep AS
        SELECT 
            id,
            "12m_call_history",
            ann_prm_amt,
            home_lot_sq_footage,
            household_policy_counts,
            newest_veh_age,
            tenure_at_snapshot,
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
    knn_train = conn.execute("""
        SELECT * FROM knn_prep
        WHERE acq_method_encoded IS NOT NULL
        ORDER BY id;
    """).fetch_df()
    knn_test = conn.execute("""
        SELECT * FROM knn_prep
        WHERE acq_method_encoded IS NULL
        ORDER BY id;
    """).fetch_df()

    X_train_knn = knn_train.drop(["acq_method_encoded", "id"], axis=1)
    y_train_knn = knn_train[["id", "acq_method_encoded"]].copy()

    X_test_knn = knn_test.drop(["acq_method_encoded", "id"], axis=1)
    y_test_knn = knn_test[["id", "acq_method_encoded"]].copy()

    if scaler is None:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_knn)
    else:
        X_train_scaled = scaler.transform(X_train_knn)
    X_test_scaled = scaler.transform(X_test_knn)

    conn.unregister("data")
    conn.execute("""DROP TABLE knn_prep;""")
    return X_train_scaled, y_train_knn, X_test_scaled, y_test_knn, scaler


def impute_df(conn, data, acq_method_imputed):
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
    ON a.id = b.id)

    SELECT * EXCLUDE (acq_method)
    FROM cte;
    """
    result = conn.execute(query).fetch_df()
    conn.unregister("data")
    conn.unregister("acq_method_imputed")

    return result


def apply_acq_method_imputation(conn, data, knn_imputer, scaler):
    """Run KNN prediction on missing acq_method rows and return data with acq_method_filled."""
    _, y_train_knn, X_test_knn, y_test_knn, _ = knn_prep(conn, data, scaler=scaler)
    if len(X_test_knn) > 0:
        y_test_knn["acq_method_encoded"] = knn_imputer.predict(X_test_knn)
    acq_method_imputed = pd.concat([y_train_knn, y_test_knn], axis=0)
    return impute_df(conn, data, acq_method_imputed)


# Fit scaler + KNN on train_split, apply to val_split and test
X_train_knn, y_train_knn, X_test_knn, y_test_knn, acq_scaler = knn_prep(conn, train_split)

knn_imputer = KNeighborsClassifier(n_neighbors=5)
knn_imputer.fit(X_train_knn, y_train_knn["acq_method_encoded"].values)

if len(X_test_knn) > 0:
    y_test_knn["acq_method_encoded"] = knn_imputer.predict(X_test_knn)
acq_method_imputed = pd.concat([y_train_knn, y_test_knn], axis=0)
train_split = impute_df(conn, train_split, acq_method_imputed)

val_split = apply_acq_method_imputation(conn, val_split, knn_imputer, acq_scaler)
test = apply_acq_method_imputation(conn, test, knn_imputer, acq_scaler)

# Step 4: Recombine train splits and export
train_imputed = pd.concat([train_split, val_split], axis=0)

save_df(conn, train_imputed, "auto_train_imputed", add_id=False)
save_df(conn, test, "auto_test_imputed", add_id=False)

print(conn.execute("SHOW TABLES").fetchall())
conn.close()
