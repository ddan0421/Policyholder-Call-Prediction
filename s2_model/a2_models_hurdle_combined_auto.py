import os
import pickle
import warnings

import duckdb
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from s1_data.a0_setup_directories import auto_model_dir
from s1_data.db_utils import load_df
from s1_data.transform_utils import apply_transformer, load_transformer
from s3_validation.model_evaluation import normalized_gini

warnings.filterwarnings("ignore")

"""
Hurdle final prediction for Auto segment.

Combines stage 1 (logistic, P(Y > 0 | X)) with stage 2 (Negative Binomial on
the positive-count subset, E[Y | Y > 0, X]) to produce hurdle predictions on
the validation set:

    y_hat(X) = P(Y > 0 | X) * E[Y | Y > 0, X]

Both models and their feature transformers (one-hot + scaler) are loaded from
auto_models/. The same raw val rows go through each transformer separately
because each stage was standardized against a different sample (full train
vs. positive-only train), so their fitted means and SDs differ.

Evaluated against actual call_counts (full y, including zeros) using RMSE
and normalized Gini.
"""

base_folder = "data"
database = "TravelersPolicyHolderCall.duckdb"
database_path = os.path.join(base_folder, database)


# Step 1: Load both stage models and their feature transformers.
with open(os.path.join(auto_model_dir, "lr_model_auto.pkl"), "rb") as f:
    lr_model = pickle.load(f)
with open(os.path.join(auto_model_dir, "nb_model_auto.pkl"), "rb") as f:
    nb_model = pickle.load(f)

binary_transformer = load_transformer(
    os.path.join(auto_model_dir, "auto_binary_transformer.pkl")
)
count_transformer = load_transformer(
    os.path.join(auto_model_dir, "auto_count_nb_transformer.pkl")
)


# Step 2: Recover the validation rows by re-running the same train_test_split
# (random_state=42) used in the stage-model data prep -- guaranteed to land on
# the exact rows the models never saw.
# Auto_train_binary supplies the SQL-transformed features; Auto_train_imputed
# supplies the actual call_counts target. Drop nonzero_call via SQL EXCLUDE
# since the hurdle product is evaluated on call_counts directly.
conn = duckdb.connect(database=database_path, read_only=True)
X_raw = load_df(conn, "Auto_train_binary", exclude_cols=["nonzero_call"])
y_raw = load_df(conn, "Auto_train_imputed")[["id", "call_counts"]]
conn.close()

_, X_val_raw, _, y_val_meta = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)
y_val = y_val_meta["call_counts"].to_numpy()


# Step 3: Apply each stage's transformer to the SAME raw val rows.
# The transformers were fit with `id` still in the input frame, so it ends up
# in feature_columns. Drop it before scoring -- both models were trained with
# id excluded (load_df(..., exclude_cols=["id"]) at training time).
def transform_and_constant(df_raw, transformer):
    X = apply_transformer(df_raw, transformer)
    X = X.drop(columns=["id"])
    return sm.add_constant(X, has_constant="add")


X_val_binary = transform_and_constant(X_val_raw, binary_transformer)
X_val_count = transform_and_constant(X_val_raw, count_transformer)


# Step 4: Stage 1 + Stage 2 predictions and the hurdle product.
prob_y_gt_0 = np.asarray(lr_model.predict(X_val_binary))
mu_y_given_pos = np.asarray(nb_model.predict(X_val_count))
y_hat = prob_y_gt_0 * mu_y_given_pos


# Step 5: Score against actual call_counts (full y, including zeros).
rmse = root_mean_squared_error(y_val, y_hat)
ng = normalized_gini(y_val, y_hat)

print("=" * 80)
print("Hurdle final prediction (Auto val)")
print("=" * 80)
print(f"  n rows                : {len(y_val)}")
print(f"  P(Y > 0)         range: [{prob_y_gt_0.min():.4f}, {prob_y_gt_0.max():.4f}]")
print(f"  E[Y | Y > 0]     range: [{mu_y_given_pos.min():.4f}, {mu_y_given_pos.max():.4f}]")
print(f"  y_hat            range: [{y_hat.min():.4f}, {y_hat.max():.4f}]")
print()
print(f"  Hurdle val RMSE             : {rmse:.4f}")
print(f"  Hurdle val NormalizedGini   : {ng:.4f}")
