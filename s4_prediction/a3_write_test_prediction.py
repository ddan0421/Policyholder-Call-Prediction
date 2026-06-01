import os
import pickle
import warnings

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm

from s1_data.a0_setup_directories import auto_model_dir, non_auto_model_dir
from s1_data.db_utils import load_df

warnings.filterwarnings("ignore")

"""
Write hurdle predictions on the holdout test set for both segments.

For each segment, load the already-encoded test tables built in
s1_data/a5_modeldat_hurdle_*:
    - test_{segment}_binary    (encoded by {segment}_binary_transformer)
    - test_{segment}_count_nb  (encoded by {segment}_count_nb_transformer)

Score with the FINAL models (refit on train + val) from
a1_final_hurdle_stage1_logit.py and a2_final_hurdle_stage2_nb.py.

    y_hat(X) = P(Y > 0 | X) * E[Y | Y > 0, X]

Writes data/test_predictions.csv with columns: id, call_counts (predicted).
Sorted by id ascending.
"""

base_folder = "data"
database = "TravelersPolicyHolderCall.duckdb"
database_path = os.path.join(base_folder, database)

output_csv = os.path.join(base_folder, "test_predictions.csv")

SEGMENTS = [
    ("auto", auto_model_dir),
    ("nonauto", non_auto_model_dir),
]


def _load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _add_const(X):
    return sm.add_constant(X, has_constant="add")


pred_parts = []

for segment, model_dir in SEGMENTS:
    print("=" * 80)
    print(f"Test prediction ({segment})")
    print("=" * 80)

    lr_model = _load_pkl(os.path.join(model_dir, f"final_lr_model_{segment}.pkl"))
    nb_model = _load_pkl(os.path.join(model_dir, f"final_nb_model_{segment}.pkl"))

    conn = duckdb.connect(database=database_path, read_only=True)
    test_binary = load_df(conn, f"test_{segment}_binary")
    test_count = load_df(conn, f"test_{segment}_count_nb")
    conn.close()

    ids = test_binary["id"].to_numpy()

    X_binary = _add_const(test_binary.drop(columns=["id"]))
    X_count = _add_const(test_count.drop(columns=["id"]))

    prob_y_gt_0 = np.asarray(lr_model.predict(X_binary))
    mu_y_given_pos = np.asarray(nb_model.predict(X_count))
    y_hat = prob_y_gt_0 * mu_y_given_pos

    print(f"  n rows                : {len(ids)}")
    print(f"  P(Y > 0)         range: [{prob_y_gt_0.min():.4f}, {prob_y_gt_0.max():.4f}]")
    print(f"  E[Y | Y > 0]     range: [{mu_y_given_pos.min():.4f}, {mu_y_given_pos.max():.4f}]")
    print(f"  y_hat            range: [{y_hat.min():.4f}, {y_hat.max():.4f}]")

    pred_parts.append(pd.DataFrame({
        "id": ids,
        "call_counts": y_hat,
    }))

predictions = pd.concat(pred_parts, axis=0, ignore_index=True).sort_values("id").reset_index(drop=True)

print()
print("=" * 80)
print("Test prediction summary (Auto + NonAuto)")
print("=" * 80)
print(f"  total rows: {len(predictions)}")
print(f"  unique ids: {predictions["id"].nunique()}")
print(f"  y_hat range: [{predictions["call_counts"].min():.4f}, {predictions["call_counts"].max():.4f}]")
print(f"  y_hat mean : {predictions["call_counts"].mean():.4f}")

predictions[["id", "call_counts"]].to_csv(output_csv, index=False)
print(f"\nTest predictions written to {output_csv}")
