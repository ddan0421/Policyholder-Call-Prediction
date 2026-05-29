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
