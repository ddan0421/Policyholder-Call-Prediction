import os
import pickle
import warnings

import duckdb
import statsmodels.api as sm
from sklearn.metrics import log_loss, roc_auc_score

from s1_data.a0_setup_directories import auto_model_dir, non_auto_model_dir
from s1_data.db_utils import load_df
from s2_model.models import sm_logit

warnings.filterwarnings("ignore")

"""
Hurdle stage 1 for both Auto and NonAuto segments.

Fit a logistic regression on (call_counts > 0) using the binary data prepared
in s1_data/a5_modeldat_hurdle_{segment}.py. The fitted probability is used as
P(Y > 0 | X). Combined with stage 2 from a3_models_hurdle_stage2_nb_auto.py:

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
    print(f"Hurdle stage 1 logit ({segment})")
    print("=" * 80)

    conn = duckdb.connect(database=database_path, read_only=True)

    # Step 1: Load model-ready tables built in s1_data/a5_modeldat_hurdle_{segment}.py
    X_train = load_df(conn, f"X_train_{segment}_binary", exclude_cols=["id"])
    X_val = load_df(conn, f"X_val_{segment}_binary", exclude_cols=["id"])
    y_train = load_df(conn, f"y_train_{segment}_binary", exclude_cols=["id"])["nonzero_call"].to_numpy()
    y_val = load_df(conn, f"y_val_{segment}_binary", exclude_cols=["id"])["nonzero_call"].to_numpy()

    conn.close()

    # Step 2: Add intercept
    X_train_const = sm.add_constant(X_train, has_constant="add")
    X_val_const = sm.add_constant(X_val, has_constant="add")

    # Step 3: Logistic Regression
    lr_model = sm_logit(X_train_const, y_train, model_output_dir=model_dir, is_scaled=True)
    lr_val_pred = lr_model.predict(X_val_const)

    log_loss_test = log_loss(y_val, lr_val_pred)
    roc_auc_test = roc_auc_score(y_val, lr_val_pred)
    print(f"Val Log-Loss: {log_loss_test:.4f}")
    print(f"Val ROC-AUC: {roc_auc_test:.4f}")

    pkl_path = os.path.join(model_dir, f"lr_model_{segment}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(lr_model, f)
    print(f"Logistic Regression model saved to {pkl_path}")
