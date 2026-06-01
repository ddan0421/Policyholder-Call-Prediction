import os
import pickle
import warnings

import duckdb
import pandas as pd
import statsmodels.api as sm

from s1_data.a0_setup_directories import auto_model_dir, non_auto_model_dir
from s1_data.db_utils import load_df
from s2_model.models import sm_truncated_nb
from s3_validation.model_evaluation import statsmodel_report

warnings.filterwarnings("ignore")

"""
Final hurdle stage 2 for both Auto and NonAuto segments.

Refit a zero-truncated Negative Binomial regression on the FULL positive-
count training data (train + val concatenated). Both X_train_*_count_nb
and X_val_*_count_nb are already filtered to call_counts > 0 in
s1_data/a5_modeldat_hurdle_*, so the concatenation is also positive-only.

E[Y | Y > 0, X] -> combined with stage 1 (a1_final_hurdle_stage1_logit.py)
into:
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
    print(f"Final hurdle stage 2 truncated NB ({segment}) -- train + val")
    print("=" * 80)

    conn = duckdb.connect(database=database_path, read_only=True)

    X_train = load_df(conn, f"X_train_{segment}_count_nb", exclude_cols=["id"])
    X_val = load_df(conn, f"X_val_{segment}_count_nb", exclude_cols=["id"])
    y_train = load_df(conn, f"y_train_{segment}_count_nb", exclude_cols=["id"])["call_counts"].to_numpy()
    y_val = load_df(conn, f"y_val_{segment}_count_nb", exclude_cols=["id"])["call_counts"].to_numpy()

    conn.close()

    X_full = pd.concat([X_train, X_val], axis=0, ignore_index=True)
    y_full = pd.concat([pd.Series(y_train), pd.Series(y_val)], axis=0, ignore_index=True).to_numpy()
    print(f"  train rows: {len(y_train)}, val rows: {len(y_val)}, combined: {len(y_full)}")

    X_full_const = sm.add_constant(X_full, has_constant="add")

    nb_model = sm_truncated_nb(X_full_const, y_full, model_output_dir=model_dir)

    nb_full_pred = nb_model.predict(X_full_const)
    statsmodel_report("Truncated NB train+val", y_full, nb_full_pred, nb_model.aic)

    pkl_path = os.path.join(model_dir, f"final_nb_model_{segment}.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(nb_model, f)
    print(f"Final zero-truncated negative binomial model saved to {pkl_path}")
