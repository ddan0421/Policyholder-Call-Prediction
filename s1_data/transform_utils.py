"""
Reusable feature-transformation helpers for the hurdle / baseline pipelines.

A "transformer" is a plain dict that captures everything needed to convert a
raw feature DataFrame into the model-ready X used at training time. By
saving it next to the trained model (pickle), we can apply identical
transforms to any new data later (validation rows, the holdout test set,
brand-new policies, etc.).

Transformer schema
------------------
{
    "nominal_cat":     [...],     # columns to one-hot encode
    "drop_first":      bool,      # passed straight to pd.get_dummies
    "feature_columns": [...],     # final ordered column list (post one-hot + reindex)
    "numeric_cols":    [...],     # columns to standardize (may be [] if standardize=False)
    "scaler":          fitted sklearn StandardScaler, or None
}

Typical usage in a training script
----------------------------------
    X_train_encoded, transformer = fit_transformer(
        X_train,
        nominal_cat=NOMINAL_CAT,
        drop_first=True,
        numeric_cols=NUMERIC_COLS,
        standardize=True,
    )
    X_val_encoded = apply_transformer(X_val, transformer)
    test_encoded = apply_transformer(test_raw, transformer)
    save_transformer(transformer, "auto_models/auto_binary_transformer.pkl")

And later at scoring / validation time
--------------------------------------
    transformer = load_transformer("auto_models/auto_binary_transformer.pkl")
    X = apply_transformer(raw_df, transformer)
    y_hat = model.predict(sm.add_constant(X, has_constant="add"))
"""

import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_transformer(X_train, nominal_cat, drop_first=True, numeric_cols=None, standardize=True):
    """
    Fit a transformer on X_train and return both (X_train_encoded, transformer).

    The fitted transformer captures the column order produced by get_dummies
    on X_train (so val/test reindex to the same set) and the StandardScaler
    fitted on X_train's numeric columns (so val/test get scaled with training
    statistics).
    """
    nominal_cat = list(nominal_cat)
    numeric_cols = list(numeric_cols) if numeric_cols else []

    X_encoded = pd.get_dummies(
        X_train, columns=nominal_cat, drop_first=drop_first, dtype="int8"
    )
    feature_columns = X_encoded.columns.tolist()

    scaler = None
    if standardize and numeric_cols:
        scaler = StandardScaler()
        X_encoded[numeric_cols] = scaler.fit_transform(X_encoded[numeric_cols])

    transformer = {
        "nominal_cat": nominal_cat,
        "drop_first": drop_first,
        "feature_columns": feature_columns,
        "numeric_cols": numeric_cols,
        "scaler": scaler,
    }
    return X_encoded, transformer


def apply_transformer(df, transformer):
    """
    Apply the same one-hot + reindex + scaling that produced X_train.

    Categories not seen at training time are silently dropped (because
    reindex keeps only feature_columns); columns seen at training but
    missing in df are filled with 0.
    """
    encoded = pd.get_dummies(
        df,
        columns=transformer["nominal_cat"],
        drop_first=transformer["drop_first"],
        dtype="int8",
    )
    encoded = encoded.reindex(columns=transformer["feature_columns"], fill_value=0)

    scaler = transformer.get("scaler")
    numeric_cols = transformer.get("numeric_cols") or []
    if scaler is not None and numeric_cols:
        encoded[numeric_cols] = scaler.transform(encoded[numeric_cols])

    return encoded


def save_transformer(transformer, path):
    with open(path, "wb") as f:
        pickle.dump(transformer, f)
    print(f"Transformer saved to {path}")


def load_transformer(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# =====================================================================
# DuckDB SQL feature transformations (cap + log1p + binarize)
#
# Each function takes an in-memory imputed DataFrame, runs the same SQL
# transformation that s1_data/a5_modeldat_hurdle_*.py applies, and returns
# the transformed DataFrame -- no DuckDB table is created.
#
# Pass `has_target=True` when the input has the `call_counts` column:
#   - binary stage adds `nonzero_call = IF(call_counts > 0, 1, 0)`
#   - count stage  passes `call_counts` through unchanged
# Pass `has_target=False` for inputs without `call_counts`.
#
# These functions never filter rows -- they only transform features. Filtering
# to call_counts > 0 for stage 2 training is the caller's responsibility.
# =====================================================================


def _run_sql_transform(conn, df, sql):
    """Register df, run sql against `input_df`, fetch result, unregister."""
    conn.register("input_df", df)
    try:
        return conn.execute(sql).fetch_df()
    finally:
        conn.unregister("input_df")


def transform_binary_auto(conn, df, has_target=True):
    """Auto hurdle stage 1 (binary classification) feature transform."""
    target_inner = ", IF(call_counts > 0, 1, 0) AS nonzero_call" if has_target else ""
    target_outer = ", nonzero_call AS nonzero_call" if has_target else ""
    return _run_sql_transform(conn, df, f"""
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
                , LEAST(household_policy_counts, 2) AS _household_policy_counts_le2
                , GREATEST(LEAST(household_policy_counts, 4) - 2, 0) AS _household_policy_counts_gt2
                , newest_veh_age
                , pol_edeliv_ind_filled
                , prdct_sbtyp_grp
                , product_sbtyp
                , telematics_ind
                , LOG(1 + LEAST(tenure_at_snapshot, 500)) AS _tenure_at_snapshot
                {target_inner}
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
            , _household_policy_counts_le2 AS household_policy_counts_le2
            , _household_policy_counts_gt2 AS household_policy_counts_gt2
            , newest_veh_age AS newest_veh_age
            , pol_edeliv_ind_filled AS pol_edeliv_ind
            , prdct_sbtyp_grp AS prdct_sbtyp_grp
            , product_sbtyp AS product_sbtyp
            , telematics_ind AS telematics_ind
            , _tenure_at_snapshot AS tenure_at_snapshot
            {target_outer}
        FROM cte
    """)


def transform_count_auto(conn, df, has_target=True):
    """Auto hurdle stage 2 (zero-truncated count) feature transform."""
    target_inner = ", call_counts" if has_target else ""
    target_outer = ", call_counts AS call_counts" if has_target else ""
    return _run_sql_transform(conn, df, f"""
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
                , LEAST(household_policy_counts, 2) AS _household_policy_counts_le2
                , GREATEST(LEAST(household_policy_counts, 4) - 2, 0) AS _household_policy_counts_gt2
                , newest_veh_age
                , pol_edeliv_ind_filled
                , prdct_sbtyp_grp
                , product_sbtyp
                , telematics_ind
                , LOG(1 + LEAST(tenure_at_snapshot, 500)) AS _tenure_at_snapshot
                {target_inner}
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
            , _household_policy_counts_le2 AS household_policy_counts_le2
            , _household_policy_counts_gt2 AS household_policy_counts_gt2
            , newest_veh_age AS newest_veh_age
            , pol_edeliv_ind_filled AS pol_edeliv_ind
            , prdct_sbtyp_grp AS prdct_sbtyp_grp
            , product_sbtyp AS product_sbtyp
            , telematics_ind AS telematics_ind
            , _tenure_at_snapshot AS tenure_at_snapshot
            {target_outer}
        FROM cte
    """)


def transform_binary_nonauto(conn, df, has_target=True):
    """NonAuto hurdle stage 1 (binary classification) feature transform."""
    target_inner = ", IF(call_counts > 0, 1, 0) AS nonzero_call" if has_target else ""
    target_outer = ", nonzero_call AS nonzero_call" if has_target else ""
    return _run_sql_transform(conn, df, f"""
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
                {target_inner}
            FROM input_df
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
            {target_outer}
        FROM cte
    """)


def transform_count_nonauto(conn, df, has_target=True):
    """NonAuto hurdle stage 2 (zero-truncated count) feature transform."""
    target_inner = ", call_counts" if has_target else ""
    target_outer = ", call_counts AS call_counts" if has_target else ""
    return _run_sql_transform(conn, df, f"""
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
                , LEAST(household_policy_counts, 3) AS _household_policy_counts_le3
                , GREATEST(LEAST(household_policy_counts, 5) - 3, 0) AS _household_policy_counts_gt3
                , pol_edeliv_ind_filled
                , prdct_sbtyp_grp
                , product_sbtyp
                , LOG(1 + LEAST(tenure_at_snapshot, 300)) AS _tenure_at_snapshot_le300
                , LOG(1 + GREATEST(LEAST(tenure_at_snapshot, 500) - 300, 0)) AS _tenure_at_snapshot_gt300
                , IF(trm_len_mo = 12, 1, 0) AS _trm_len_mo
                {target_inner}
            FROM input_df
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
            , _household_policy_counts_le3 AS household_policy_counts_le3
            , _household_policy_counts_gt3 AS household_policy_counts_gt3
            , pol_edeliv_ind_filled AS pol_edeliv_ind
            , prdct_sbtyp_grp AS prdct_sbtyp_grp
            , product_sbtyp AS product_sbtyp
            , _tenure_at_snapshot_le300 AS tenure_at_snapshot_le300
            , _tenure_at_snapshot_gt300 AS tenure_at_snapshot_gt300
            , _trm_len_mo AS trm_len_mo
            {target_outer}
        FROM cte
    """)
