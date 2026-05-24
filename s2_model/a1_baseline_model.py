import os
import warnings

import duckdb
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import root_mean_squared_error

from s1_data.db_utils import load_df

warnings.filterwarnings("ignore")

"""
Baseline zero-inflated count models for Auto segment (statsmodels):
1. Zero-Inflated Poisson (ZIP)
2. Zero-Inflated Negative Binomial (ZINB)

References:
- ZIP:  https://www.statsmodels.org/dev/generated/statsmodels.discrete.count_model.ZeroInflatedPoisson.html
- ZINB: https://www.statsmodels.org/dev/generated/statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP.html
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

# Step 2: Build design matrices (cast, align, add intercept)
X_train = X_train.astype(float)
X_val = X_val.reindex(columns=X_train.columns, fill_value=0).astype(float)

X_train_const = sm.add_constant(X_train, has_constant="add")
X_val_const = sm.add_constant(X_val, has_constant="add")

# Intercept-only inflation: avoids over-parameterized ZI logit and helps convergence
infl_train = np.ones((len(y_train), 1))
infl_val = np.ones((len(y_val), 1))


# Step 3: Metrics
def relative_gini(y_true, y_pred):
    """Relative Gini (README): rank rows by predicted call counts (descending)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    order = np.argsort(-y_pred, kind="mergesort")
    a = y_true[order]
    n = len(a)
    k = np.arange(1, n + 1, dtype=float)
    numer = np.sum(k * a - np.cumsum(a))
    denom = a.sum() * k.sum()
    return float(numer / denom) if denom > 0 else float("nan")


def evaluate(label, result, exog, exog_infl, y_true):
    y_pred = result.predict(exog=exog, exog_infl=exog_infl, which="mean")
    y_true_arr = np.asarray(y_true)
    rmse = root_mean_squared_error(y_true_arr, y_pred)
    gini = relative_gini(y_true_arr, y_pred)
    aic = getattr(result, "aic", float("nan"))
    print(f"{label}: RMSE={rmse:.4f}  AIC={aic:.2f}  RelativeGini={gini:.4f}")
    return y_pred


# Sanity-check: plain Poisson GLM (also reused as warm-start for ZIP/ZINB)
print("Fitting Poisson GLM (warm-start + diagnostic)...")
pois_glm = sm.GLM(y_train, X_train_const.values, family=sm.families.Poisson()).fit()
pois_start = pois_glm.params

pois_train_pred = pois_glm.predict(X_train_const.values)
pois_val_pred = pois_glm.predict(X_val_const.values)
print(
    f"Poisson GLM train: RMSE={root_mean_squared_error(y_train, pois_train_pred):.4f}  "
    f"RelativeGini={relative_gini(y_train, pois_train_pred):.4f}  "
    f"pred std={pois_train_pred.std():.4f}"
)
print(
    f"Poisson GLM val  : RMSE={root_mean_squared_error(y_val, pois_val_pred):.4f}  "
    f"RelativeGini={relative_gini(y_val, pois_val_pred):.4f}  "
    f"pred std={pois_val_pred.std():.4f}"
)
print(f"|coef| max={np.abs(pois_glm.params[1:]).max():.4f}, "
      f"mean={np.abs(pois_glm.params[1:]).mean():.4f} (excluding intercept)")


# Step 4: Zero-Inflated Poisson (ZIP)
print("=" * 80)
print("Fitting Zero-Inflated Poisson (ZIP)")
print("=" * 80)
zip_start = np.r_[0.0, pois_start]
zip_model = sm.ZeroInflatedPoisson(
    endog=y_train,
    exog=X_train_const.values,
    exog_infl=infl_train,
    inflation="logit",
)
zip_result = zip_model.fit(start_params=zip_start, method="lbfgs", maxiter=2000, disp=False)
print(zip_result.summary())
evaluate("ZIP train", zip_result, X_train_const.values, infl_train, y_train)
evaluate("ZIP val  ", zip_result, X_val_const.values, infl_val, y_val)


# Step 5: Zero-Inflated Negative Binomial (ZINB)
print("=" * 80)
print("Fitting Zero-Inflated Negative Binomial (ZINB)")
print("=" * 80)
zinb_start = np.r_[0.0, pois_start, 1.0]
zinb_model = sm.ZeroInflatedNegativeBinomialP(
    endog=y_train,
    exog=X_train_const.values,
    exog_infl=infl_train,
    inflation="logit",
    p=2,
)
zinb_result = zinb_model.fit(start_params=zinb_start, method="lbfgs", maxiter=2000, disp=False)
print(zinb_result.summary())
evaluate("ZINB train", zinb_result, X_train_const.values, infl_train, y_train)
evaluate("ZINB val  ", zinb_result, X_val_const.values, infl_val, y_val)
