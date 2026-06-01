import os
import pickle
import warnings

import duckdb
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import log_loss, roc_auc_score

from s1_data.a0_setup_directories import auto_model_dir, non_auto_model_dir
from s1_data.db_utils import load_df
from s2_model.models import sm_logit

warnings.filterwarnings("ignore")

"""
Final hurdle stage 1 for both Auto and NonAuto segments.

Refit a logistic regression on the FULL labeled training data (train + val
concatenated) using the binary tables prepared in s1_data/a5_modeldat_hurdle_*.
Hyperparameters / feature spec were already chosen on train-only +
val-holdout in s2_model; here we just fit the production model on all
available labels for test scoring.

P(Y > 0 | X) -> combined with stage 2 (a2_final_hurdle_stage2_nb.py) into:
    y_hat(X) = P(Y > 0 | X) * E[Y | Y > 0, X]
"""

base_folder = "data"
database = "TravelersPolicyHolderCall.duckdb"
database_path = os.path.join(base_folder, database)

SEGMENTS = [
    ("auto", auto_model_dir),
    ("nonauto", non_auto_model_dir),
]


for segment, model_dir in SEGMENTS:
    print("=" * 80)
    print(f"Final hurdle stage 1 logit ({segment}) -- train + val")
    print("=" * 80)

    conn = duckdb.connect(database=database_path, read_only=True)

    X_train = load_df(conn, f"X_train_{segment}_binary", exclude_cols=["id"])
    X_val = load_df(conn, f"X_val_{segment}_binary", exclude_cols=["id"])
    y_train = load_df(conn, f"y_train_{segment}_binary", exclude_cols=["id"])["nonzero_call"].to_numpy()
    y_val = load_df(conn, f"y_val_{segment}_binary", exclude_cols=["id"])["nonzero_call"].to_numpy()

    conn.close()

    X_full = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_full = pd.concat([pd.Series(y_train), pd.Series(y_val)], axis=0, ignore_index=True).to_numpy()
    print(f"  train rows: {len(y_train)}, val rows: {len(y_val)}, combined: {len(y_full)}")

    X_full_const = sm.add_constant(X_full, has_constant="add")

    lr_model = sm_logit(X_full_const, y_full, model_output_dir=model_dir, is_scaled=True)

    lr_full_pred = lr_model.predict(X_full_const)
    print(f"In-sample (train+val) Log-Loss: {log_loss(y_full, lr_full_pred):.4f}")
    print(f"In-sample (train+val) ROC-AUC : {roc_auc_score(y_full, lr_full_pred):.4f}")

    pkl_path = os.path.join(model_dir, f"final_lr_model_{segment}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(lr_model, f)
    print(f"Final logistic regression model saved to {pkl_path}")
