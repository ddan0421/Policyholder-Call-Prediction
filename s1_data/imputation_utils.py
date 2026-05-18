"""
Shared KNN imputation helpers for pol_edeliv_ind (-2 = missing).

Used by both a2_data_prep_auto.py and a2_data_prep_nonauto.py.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def knn_prep(conn, data, scaler=None):
    """
    Build features for KNN imputation of pol_edeliv_ind.

    Encodes pol_edeliv_ind: 0/1/-1 stay as-is, -2 becomes NULL (target to impute).
    Splits rows into knn_train (label known) and knn_test (label = -2 / NULL).
    One-hot encodes geo_group with fixed categories so train/val/test always
    produce the same dummy columns in the same order, even if a split happens
    to be missing one of the three values. Scales continuous numerics with the
    provided scaler (fit on train when scaler is None).
    """
    conn.register("data", data)

    query = """
    CREATE OR REPLACE TABLE knn_prep AS
        SELECT
            id,
            "12m_call_history",
            ann_prm_amt,
            household_policy_counts,
            tenure_at_snapshot,
            digital_contact_ind,
            has_prior_carrier,
            geo_group,
            CAST(CASE
                WHEN pol_edeliv_ind = 0 THEN 0
                WHEN pol_edeliv_ind = 1 THEN 1
                WHEN pol_edeliv_ind = -1 THEN -1
                ELSE NULL
            END AS INTEGER) AS pol_edeliv_ind_encoded
        FROM data;
    """
    conn.execute(query)
    knn_train = conn.execute("""
        SELECT * FROM knn_prep
        WHERE pol_edeliv_ind_encoded IS NOT NULL
        ORDER BY id;
    """).fetch_df()
    knn_test = conn.execute("""
        SELECT * FROM knn_prep
        WHERE pol_edeliv_ind_encoded IS NULL
        ORDER BY id;
    """).fetch_df()

    geo_categories = ["rural", "suburban", "urban"]
    knn_train["geo_group"] = pd.Categorical(knn_train["geo_group"], categories=geo_categories)
    knn_test["geo_group"]  = pd.Categorical(knn_test["geo_group"],  categories=geo_categories)
    knn_train = pd.get_dummies(knn_train, columns=["geo_group"], prefix="geo", dtype="int8")
    knn_test  = pd.get_dummies(knn_test,  columns=["geo_group"], prefix="geo", dtype="int8")

    X_train_knn = knn_train.drop(["pol_edeliv_ind_encoded", "id"], axis=1).copy()
    y_train_knn = knn_train[["id", "pol_edeliv_ind_encoded"]].copy()

    X_test_knn = knn_test.drop(["pol_edeliv_ind_encoded", "id"], axis=1).copy()
    y_test_knn = knn_test[["id", "pol_edeliv_ind_encoded"]].copy()

    numeric_cols = ["12m_call_history", "ann_prm_amt",
                    "household_policy_counts", "tenure_at_snapshot"]

    if scaler is None:
        scaler = StandardScaler()
        X_train_knn[numeric_cols] = scaler.fit_transform(X_train_knn[numeric_cols])
    else:
        X_train_knn[numeric_cols] = scaler.transform(X_train_knn[numeric_cols])
    X_test_knn[numeric_cols] = scaler.transform(X_test_knn[numeric_cols])

    conn.unregister("data")
    conn.execute("""DROP TABLE knn_prep;""")
    return X_train_knn, y_train_knn, X_test_knn, y_test_knn, scaler


def impute_df(conn, data, pol_edeliv_ind_imputed):
    """Replace pol_edeliv_ind = -2 rows with the imputed values, joined on id."""
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
    ON a.id = b.id)

    SELECT * EXCLUDE (pol_edeliv_ind)
    FROM cte;
    """
    result = conn.execute(query).fetch_df()
    conn.unregister("data")
    conn.unregister("pol_edeliv_ind_imputed")

    return result


def apply_imputation(conn, data, knn_imputer, scaler):
    """Apply an already-fitted KNN imputer + scaler to a new dataframe."""
    _, y_train_knn, X_test_knn, y_test_knn, _ = knn_prep(conn, data, scaler=scaler)
    if len(X_test_knn) > 0:
        y_test_knn["pol_edeliv_ind_encoded"] = knn_imputer.predict(X_test_knn)
    pol_edeliv_ind_imputed = pd.concat([y_train_knn, y_test_knn], axis=0)
    return impute_df(conn, data, pol_edeliv_ind_imputed)


def fit_and_impute_train(conn, train_split, n_neighbors=5):
    """
    Fit scaler + KNN classifier on train_split, impute its own pol_edeliv_ind,
    and return (imputed_train_split, fitted_imputer, fitted_scaler) so callers
    can apply the fitted objects to validation and test data.
    """
    X_train_knn, y_train_knn, X_test_knn, y_test_knn, scaler = knn_prep(conn, train_split)

    imputer = KNeighborsClassifier(n_neighbors=n_neighbors)
    imputer.fit(X_train_knn, y_train_knn["pol_edeliv_ind_encoded"].values)

    if len(X_test_knn) > 0:
        y_test_knn["pol_edeliv_ind_encoded"] = imputer.predict(X_test_knn)
    pol_edeliv_ind_imputed = pd.concat([y_train_knn, y_test_knn], axis=0)
    train_imputed = impute_df(conn, train_split, pol_edeliv_ind_imputed)

    return train_imputed, imputer, scaler
