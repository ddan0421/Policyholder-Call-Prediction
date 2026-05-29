import os
import pickle
import warnings

import duckdb
import statsmodels.api as sm

from s1_data.a0_setup_directories import auto_model_dir
from s1_data.db_utils import load_df
from s2_model.models import sm_truncated_nb
from s3_validation.model_evaluation import statsmodel_report

warnings.filterwarnings("ignore")

"""
Hurdle stage 2 for Auto segment.

Fit a zero-truncated Negative Binomial regression on the positive-count
subset (y > 0) prepared in s1_data/a5_modeldat_hurdle_auto.py. The truncated
likelihood explicitly excludes Y = 0, so model.predict(X) returns
E[Y | Y > 0, X] directly. Combined with stage 1
(s2_model/a2_models_hurdle_logit_auto.py):

    y_hat(X) = P(Y > 0 | X) * E[Y | Y > 0, X]
"""

base_folder = "data"
database = "TravelersPolicyHolderCall.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=True)

# Step 1: Load model-ready tables built in s1_data/a5_modeldat_hurdle_auto.py.
# X_*_count_nb / y_*_count are already filtered to call_counts > 0.
X_train = load_df(conn, "X_train_auto_count_nb", exclude_cols=["id"])
X_val = load_df(conn, "X_val_auto_count_nb", exclude_cols=["id"])
y_train = load_df(conn, "y_train_auto_count", exclude_cols=["id"])["call_counts"].to_numpy()
y_val = load_df(conn, "y_val_auto_count", exclude_cols=["id"])["call_counts"].to_numpy()

conn.close()


# Step 2: Add intercept (the model function also does this defensively).
X_train_const = sm.add_constant(X_train, has_constant="add")
X_val_const = sm.add_constant(X_val, has_constant="add")


# Step 3: Zero-truncated Negative Binomial regression on the positive-count subset.
print("=" * 80)
print("Fitting zero-truncated Negative Binomial regression (hurdle stage 2)")
print("=" * 80)
nb_model = sm_truncated_nb(X_train_const, y_train, model_output_dir=auto_model_dir)

nb_train_pred = nb_model.predict(X_train_const)
nb_val_pred = nb_model.predict(X_val_const)
statsmodel_report("Truncated NB train", y_train, nb_train_pred, nb_model.aic)
statsmodel_report("Truncated NB val  ", y_val, nb_val_pred, nb_model.aic)


# Step 4: Persist for stage-2 validation / hurdle product downstream.
nb_pkl_path = os.path.join(auto_model_dir, "nb_model_auto.pkl")
with open(nb_pkl_path, "wb") as f:
    pickle.dump(nb_model, f)
print(f"Zero-truncated Negative Binomial model saved to {nb_pkl_path}")
