import os
import warnings

import duckdb
import numpy as np
import statsmodels.api as sm

from s1_data.db_utils import load_df
from s2_model.models import sm_poisson_glm, sm_zip, sm_zinb, zi_warm_start
from s3_validation.model_evaluation import statsmodel_report

warnings.filterwarnings("ignore")

"""
Baseline count models for Auto segment (statsmodels):
1. Poisson GLM (reference)
2. Zero-Inflated Poisson (ZIP) with feature-driven inflation
3. Zero-Inflated Negative Binomial (ZINB) with feature-driven inflation

"""

base_folder = "data"
database = "TravelersPolicyHolderCall.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=True)

# Step 1: Load model-ready tables built in s1_data/a4_baseline_modeldat_auto.py
X_train = load_df(conn, "X_train_base", delete_id=True)
X_val = load_df(conn, "X_val_base", delete_id=True)
y_train = load_df(conn, "y_train", delete_id=True)["call_counts"].to_numpy()
y_val = load_df(conn, "y_val", delete_id=True)["call_counts"].to_numpy()

conn.close()

# Step 2: Add intercept (model functions also do this defensively)
X_train_const = sm.add_constant(X_train, has_constant="add")
X_val_const = sm.add_constant(X_val, has_constant="add")


# Step 3: Poisson GLM
print("=" * 80)
print("Fitting Poisson GLM")
print("=" * 80)
pois_result = sm_poisson_glm(X_train_const, y_train)
pois_train_pred = pois_result.predict(X_train_const)
pois_val_pred = pois_result.predict(X_val_const)
statsmodel_report("Poisson train", y_train, pois_train_pred, pois_result.aic)
statsmodel_report("Poisson val  ", y_val, pois_val_pred, pois_result.aic)


# Step 4: Warm-start parameters for ZIP / ZINB
# - Inflation logit warm-started from logistic regression of (y == 0) on X
# - Count part warm-started from Poisson GLM on positive-count subset (y > 0)
print("Computing warm-start parameters for ZIP / ZINB...")
infl_start, count_start = zi_warm_start(X_train_const, y_train)
zip_start_params = np.r_[infl_start, count_start]
zinb_start_params = np.r_[infl_start, count_start, 1.0]


# Step 5: Zero-Inflated Poisson (ZIP)
print("=" * 80)
print("Fitting Zero-Inflated Poisson (ZIP) with feature-driven inflation")
print("=" * 80)
zip_result = sm_zip(X_train_const, y_train, start_params=zip_start_params)
zip_train_pred = zip_result.predict(exog=X_train_const, exog_infl=X_train_const, which="mean")
zip_val_pred = zip_result.predict(exog=X_val_const, exog_infl=X_val_const, which="mean")
statsmodel_report("ZIP train", y_train, zip_train_pred, zip_result.aic)
statsmodel_report("ZIP val  ", y_val, zip_val_pred, zip_result.aic)


# Step 6: Zero-Inflated Negative Binomial (ZINB)
print("=" * 80)
print("Fitting Zero-Inflated Negative Binomial (ZINB) with feature-driven inflation")
print("=" * 80)
zinb_result = sm_zinb(X_train_const, y_train, start_params=zinb_start_params)
zinb_train_pred = zinb_result.predict(exog=X_train_const, exog_infl=X_train_const, which="mean")
zinb_val_pred = zinb_result.predict(exog=X_val_const, exog_infl=X_val_const, which="mean")
statsmodel_report("ZINB train", y_train, zinb_train_pred, zinb_result.aic)
statsmodel_report("ZINB val  ", y_val, zinb_val_pred, zinb_result.aic)


# Step 7: Side-by-side validation comparison
print("=" * 80)
print("Validation comparison")
print("=" * 80)
statsmodel_report("Poisson val", y_val, pois_val_pred, pois_result.aic)
statsmodel_report("ZIP val    ", y_val, zip_val_pred, zip_result.aic)
statsmodel_report("ZINB val   ", y_val, zinb_val_pred, zinb_result.aic)
