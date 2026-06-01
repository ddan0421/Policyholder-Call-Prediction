import os
import pickle
import warnings

import duckdb
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from s1_data.a0_setup_directories import auto_model_dir, non_auto_model_dir
from s1_data.db_utils import load_df
from s1_data.transform_utils import (
    apply_transformer,
    load_transformer,
    transform_binary_auto,
    transform_binary_nonauto,
    transform_count_auto,
    transform_count_nonauto,
)
from s3_validation.model_evaluation import normalized_gini

warnings.filterwarnings("ignore")

"""
Hurdle final prediction for both Auto and NonAuto segments.

y_hat(X) = P(Y > 0 | X) * E[Y | Y > 0, X]

Per segment: split {Segment}_train_imputed (random_state=42) to recover the
val rows, run each stage's SQL feature transform on the val subset, apply the
saved sklearn transformer + model, and multiply for the hurdle product.
Evaluated against actual call_counts (full y, including zeros).
"""

base_folder = "data"
database = "TravelersPolicyHolderCall.duckdb"
database_path = os.path.join(base_folder, database)

SEGMENTS = [
    ("auto",    auto_model_dir,     "Auto",    transform_binary_auto,    transform_count_auto),
    ("nonauto", non_auto_model_dir, "NonAuto", transform_binary_nonauto, transform_count_nonauto),
]


def transform_and_constant(df_raw, transformer):
    X = apply_transformer(df_raw, transformer)
    X = X.drop(columns=["id"])
    return sm.add_constant(X, has_constant="add")


for segment, model_dir, imputed_table, sql_binary, sql_count in SEGMENTS:
    print("=" * 80)
    print(f"Hurdle final prediction ({segment} val)")
    print("=" * 80)

    with open(os.path.join(model_dir, f"lr_model_{segment}.pkl"), "rb") as f:
        lr_model = pickle.load(f)
    with open(os.path.join(model_dir, f"nb_model_{segment}.pkl"), "rb") as f:
        nb_model = pickle.load(f)

    binary_transformer = load_transformer(
        os.path.join(model_dir, f"{segment}_binary_transformer.pkl")
    )
    count_transformer = load_transformer(
        os.path.join(model_dir, f"{segment}_count_nb_transformer.pkl")
    )

    conn = duckdb.connect(database=database_path, read_only=True)
    raw = load_df(conn, f"{imputed_table}_train_imputed")
    X_raw = raw.drop(columns=["call_counts"])
    y_raw = raw[["id", "call_counts"]]

    _, X_val_raw, _, y_val_meta = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42
    )
    y_val = y_val_meta["call_counts"].to_numpy()

    # SQL feature transforms (no targets needed for scoring)
    X_val_binary_raw = sql_binary(conn, X_val_raw, has_target=False)
    X_val_count_raw = sql_count(conn, X_val_raw, has_target=False)
    conn.close()

    X_val_binary = transform_and_constant(X_val_binary_raw, binary_transformer)
    X_val_count = transform_and_constant(X_val_count_raw, count_transformer)

    prob_y_gt_0 = np.asarray(lr_model.predict(X_val_binary))
    mu_y_given_pos = np.asarray(nb_model.predict(X_val_count))
    y_hat = prob_y_gt_0 * mu_y_given_pos

    rmse = root_mean_squared_error(y_val, y_hat)
    ng = normalized_gini(y_val, y_hat)

    print(f"  n rows                : {len(y_val)}")
    print(f"  P(Y > 0)         range: [{prob_y_gt_0.min():.4f}, {prob_y_gt_0.max():.4f}]")
    print(f"  E[Y | Y > 0]     range: [{mu_y_given_pos.min():.4f}, {mu_y_given_pos.max():.4f}]")
    print(f"  y_hat            range: [{y_hat.min():.4f}, {y_hat.max():.4f}]")
    print()
    print(f"  Hurdle val RMSE             : {rmse:.4f}")
    print(f"  Hurdle val NormalizedGini   : {ng:.4f}")
    print()
