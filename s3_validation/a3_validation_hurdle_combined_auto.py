import os
import pickle
import warnings

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from s1_data.a0_setup_directories import auto_model_dir
from s1_data.db_utils import load_df
from s1_data.transform_utils import apply_transformer, load_transformer

warnings.filterwarnings("ignore")

"""
Combined hurdle calibration for Auto segment.

Same recipe as s2_model/a4_models_hurdle_combined_auto.py, but here we keep
both Train and Val splits and aggregate by predictor level so we can plot
actual vs predicted marginal mean E[Y] (zeros included). This mirrors the
EDA `plot_marginal_mean_curves` view (06_marginal_mean_*) but overlays the
hurdle product:

    y_hat(X) = P(Y > 0 | X) * E[Y | Y > 0, X]

Saved as 14_actual_vs_pred_marginal_*.
"""

base_folder = "data"
database = "TravelersPolicyHolderCall.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=True)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


lr_model = load_pkl(os.path.join(auto_model_dir, "lr_model_auto.pkl"))
nb_model = load_pkl(os.path.join(auto_model_dir, "nb_model_auto.pkl"))

binary_transformer = load_transformer(
    os.path.join(auto_model_dir, "auto_binary_transformer.pkl")
)
count_transformer = load_transformer(
    os.path.join(auto_model_dir, "auto_count_nb_transformer.pkl")
)


# Step 1: Recover both Train and Val rows with the same split used at training
# time (random_state=42, test_size=0.2). Auto_train_binary supplies the cap +
# log1p transformed features for both stages; the actual call_counts are read
# later from Auto_train_imputed via the SQL JOIN.
X_raw = load_df(conn, "Auto_train_binary", exclude_cols=["nonzero_call"])
y_raw = load_df(conn, "Auto_train_imputed")[["id", "call_counts"]]

X_train_raw, X_val_raw, _, _ = train_test_split(
    X_raw, y_raw, test_size=0.2, random_state=42
)


# Step 2: Apply each stage's transformer separately (each was standardized
# against a different sample, so their fitted means/SDs differ), then form the
# hurdle product y_hat = P(Y > 0) * E[Y | Y > 0].
def transform_and_constant(df_raw, transformer):
    X = apply_transformer(df_raw, transformer)
    X = X.drop(columns=["id"])
    return sm.add_constant(X, has_constant="add")


def hurdle_predict(df_raw):
    X_binary = transform_and_constant(df_raw, binary_transformer)
    X_count = transform_and_constant(df_raw, count_transformer)
    prob_y_gt_0 = np.asarray(lr_model.predict(X_binary))
    mu_y_given_pos = np.asarray(nb_model.predict(X_count))
    return prob_y_gt_0 * mu_y_given_pos


train_pred = hurdle_predict(X_train_raw)
val_pred = hurdle_predict(X_val_raw)


# Step 3: Build a (id, pred, Sample) table for join. We only keep what we need;
# the original features are read from Auto_train_imputed inside the SQL below.
hurdle_pred = pd.concat(
    [
        pd.DataFrame({"id": X_train_raw["id"].to_numpy(), "pred": train_pred, "Sample": "Train"}),
        pd.DataFrame({"id": X_val_raw["id"].to_numpy(),   "pred": val_pred,   "Sample": "Val"}),
    ],
    axis=0,
    ignore_index=True,
)

conn.register("hurdle_pred", hurdle_pred)


def _sort_by_valvar(df):
    """Sort numerically when ValVar is numeric-looking, alphabetically otherwise."""
    try:
        out = df.copy()
        out["_sort_key"] = out["ValVar"].astype(float)
        return out.sort_values("_sort_key").drop(columns="_sort_key")
    except ValueError:
        return df.sort_values("ValVar")


def build_valid_table(conn, varlist):
    """Aggregate actual and predicted marginal mean E[Y] per (variable, level, sample)."""
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE validTable (
            Variable    VARCHAR,
            ValVar      VARCHAR,
            Sample      VARCHAR,
            PredMean    DOUBLE,
            ActualMean  DOUBLE,
            TotalCnt    INT
        );
    """)
    for var in varlist:
        conn.execute(f"""
            INSERT INTO validTable
                SELECT
                    '{var}' AS Variable,
                    {var}::VARCHAR AS ValVar,
                    Sample,
                    AVG(pred) AS PredMean,
                    AVG(call_counts) AS ActualMean,
                    COUNT(*) AS TotalCnt
                FROM validdat
                GROUP BY 1, 2, 3
                ORDER BY 1, 2, 3
        """)
    return conn.execute(
        "SELECT * FROM validTable ORDER BY Variable, Sample, ValVar;"
    ).fetch_df()


def plot_actual_vs_pred(valid_table, varlist, segment, prefix):
    """One PNG per (variable, sample): overlay actual vs predicted marginal mean E[Y]."""
    os.makedirs("plots", exist_ok=True)
    samples = sorted(valid_table["Sample"].dropna().unique())
    for var in varlist:
        for sample in samples:
            sub = _sort_by_valvar(
                valid_table[
                    (valid_table["Variable"] == var)
                    & (valid_table["Sample"] == sample)
                ]
            )
            if sub.empty:
                print(f"[skip] {segment} {sample} {var}: no rows in validTable")
                continue

            x = np.arange(len(sub))
            labels = sub["ValVar"].astype(str)

            fig, ax1 = plt.subplots(figsize=(12, 5))
            ax1.plot(x, sub["ActualMean"], color="#C44E52", marker="o", linewidth=2,
                     label="Actual E[y]")
            ax1.plot(x, sub["PredMean"], color="#4C72B0", marker="s", linewidth=2,
                     linestyle="--", label="Predicted E[y]")
            ax1.set_xlabel(var)
            ax1.set_ylabel("Marginal mean call_counts")
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, rotation=45, ha="right")

            ax2 = ax1.twinx()
            ax2.bar(x, sub["TotalCnt"], color="#7F7F7F", alpha=0.20, label="Total count")
            ax2.set_ylabel("Count")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            fig.suptitle(
                f"{var}: actual vs predicted marginal mean E[y] ({segment} {sample})", y=1.02
            )
            plt.tight_layout()
            filename = (
                f"{prefix}actual_vs_pred_marginal_{segment.lower()}_{var}_{sample.lower()}.png"
            )
            fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] plots/{filename}")


# Step 4: Build validdat. Use original-scale features from Auto_train_imputed
# (same bins/caps as EDA section 4) joined with hurdle_pred. Both Train and
# Val rows are kept; the plotter splits them so we can see calibration on
# each split separately.
auto_varlist = [
    "binned_12m_call_history", "acq_method", "binned_ann_prm_amt", "bi_limit_group",
    "digital_contact_ind", "geo_group", "has_prior_carrier", "binned_home_lot_sq_footage",
    "household_group", "capped_household_policy_counts", "newest_veh_age",
    "pay_type_code", "pol_edeliv_ind_filled", "prdct_sbtyp_grp", "product_sbtyp",
    "telematics_ind", "binned_tenure_at_snapshot",
]

conn.execute("""
    CREATE OR REPLACE TEMP TABLE validdat AS
        SELECT
            LEAST((a."12m_call_history" / 5)::INT * 5, 40) AS binned_12m_call_history,
            a.acq_method,
            LEAST((a.ann_prm_amt / 900)::INT * 900, 7200) AS binned_ann_prm_amt,
            a.bi_limit_group,
            a.digital_contact_ind,
            a.geo_group,
            a.has_prior_carrier,
            (a.home_lot_sq_footage / 10000)::INT * 10000 AS binned_home_lot_sq_footage,
            a.household_group,
            LEAST(a.household_policy_counts, 4) AS capped_household_policy_counts,
            a.newest_veh_age,
            a.pay_type_code,
            a.pol_edeliv_ind_filled,
            a.prdct_sbtyp_grp,
            a.product_sbtyp,
            a.telematics_ind,
            LEAST((a.tenure_at_snapshot / 100)::INT * 100, 500) AS binned_tenure_at_snapshot,
            a.call_counts,
            b.Sample,
            b.pred
        FROM Auto_train_imputed AS a
        JOIN hurdle_pred AS b
        ON a.id = b.id;
""")

auto_valid_table = build_valid_table(conn, auto_varlist)
plot_actual_vs_pred(auto_valid_table, auto_varlist, "Auto", "14_")

conn.close()
