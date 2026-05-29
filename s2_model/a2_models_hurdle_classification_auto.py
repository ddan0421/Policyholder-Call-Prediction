import os
import warnings

import duckdb
import statsmodels.api as sm
import pickle

from s1_data.a0_setup_directories import auto_model_dir
from s1_data.db_utils import load_df
from s2_model.models import sm_logit

from sklearn.metrics import roc_auc_score, log_loss



warnings.filterwarnings("ignore")

"""
Baseline count models for Non Auto segment (statsmodels):
1. Poisson GLM (reference)
2. Zero-Inflated Poisson (ZIP) with feature-driven inflation
3. Zero-Inflated Negative Binomial (ZINB) with feature-driven inflation

"""

base_folder = "data"
database = "TravelersPolicyHolderCall.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=True)

# Step 1: Load model-ready tables built in s1_data/a5_modeldat_hurdle_auto.py
X_train = load_df(conn, "X_train_auto_binary", exclude_cols=["id"])
X_val = load_df(conn, "X_val_auto_binary", exclude_cols=["id"])
y_train = load_df(conn, "y_train_auto_binary", exclude_cols=["id"])["nonzero_call"].to_numpy()
y_val = load_df(conn, "y_val_auto_binary", exclude_cols=["id"])["nonzero_call"].to_numpy()



# Step 2: Add intercept (model functions also do this defensively)
X_train_const = sm.add_constant(X_train, has_constant="add")
X_val_const = sm.add_constant(X_val, has_constant="add")


# Step 3: Logistic Regression
print("=" * 80)
print("Fitting Logistic Regression")
print("=" * 80)
lr_model = sm_logit(X_train_const, y_train, model_output_dir=auto_model_dir, is_scaled=True)
lr_train_pred = lr_model.predict(X_train_const)
lr_val_pred = lr_model.predict(X_val_const)

log_loss_test = log_loss(y_val, lr_val_pred)
roc_auc_test = roc_auc_score(y_val, lr_val_pred)
print(f"Val Log-Loss: {log_loss_test:.4f}")
print(f"Val ROC-AUC: {roc_auc_test:.4f}")


with open(os.path.join(auto_model_dir, "lr_model_auto.pkl"), "wb") as f:
    pickle.dump(lr_model, f)
print(f"Logistic Regression model saved to {os.path.join(auto_model_dir, "lr_model_auto.pkl")}")


