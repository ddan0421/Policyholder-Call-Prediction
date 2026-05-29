import os
import pickle

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from s1_data.a0_setup_directories import auto_model_dir
from s1_data.db_utils import load_df

base_folder = "data"
database = "TravelersPolicyHolderCall.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=True)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


nb_model = load_pkl(os.path.join(auto_model_dir, "nb_model_auto.pkl"))


# Step 1: Score the train and val splits with the saved truncated NB model.
# Loaded with id kept (no exclude_cols) so we can attach predictions back to
# raw features in DuckDB. X_*_auto_count_nb tables are already filtered to
# call_counts > 0 by the data-prep step, so every scored row has an actual
# positive call count to compare against.
X_train = load_df(conn, "X_train_auto_count_nb")
X_val = load_df(conn, "X_val_auto_count_nb")

X_train_const = sm.add_constant(X_train.drop(columns=["id"]), has_constant="add")
X_val_const = sm.add_constant(X_val.drop(columns=["id"]), has_constant="add")

# TruncatedLFNegativeBinomialP.predict() returns E[Y | Y > 0, X] directly --
# the truncation is baked into the likelihood, so no manual correction needed.
nb_train_pred = nb_model.predict(X_train_const)
nb_val_pred = nb_model.predict(X_val_const)


# Step 2: Build a (id, pred, Sample) table for join. We only keep what we need;
# the original features are read from Auto_train_imputed inside the SQL below.
nb_pred = pd.concat(
    [
        pd.DataFrame({"id": X_train["id"].to_numpy(), "pred": nb_train_pred.to_numpy(), "Sample": "Train"}),
        pd.DataFrame({"id": X_val["id"].to_numpy(),   "pred": nb_val_pred.to_numpy(),   "Sample": "Val"}),
    ],
    axis=0,
    ignore_index=True,
)

# Register so DuckDB can JOIN against it (avoids relying on replacement scans).
conn.register("nb_pred", nb_pred)


def _sort_by_valvar(df):
    """Sort numerically when ValVar is numeric-looking, alphabetically otherwise."""
    try:
        out = df.copy()
        out["_sort_key"] = out["ValVar"].astype(float)
        return out.sort_values("_sort_key").drop(columns="_sort_key")
    except ValueError:
        return df.sort_values("ValVar")


def build_valid_table(conn, varlist):
    """Aggregate actual and predicted E[Y | Y > 0] per (variable, level, sample)."""
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
    """One PNG per (variable, sample): overlay actual vs predicted E[Y | Y > 0]."""
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
                     label="Actual E[y | y > 0]")
            ax1.plot(x, sub["PredMean"], color="#4C72B0", marker="s", linewidth=2,
                     linestyle="--", label="Predicted E[y | y > 0]")
            ax1.set_xlabel(var)
            ax1.set_ylabel("E[call_counts | call_counts > 0]")
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, rotation=45, ha="right")

            ax2 = ax1.twinx()
            ax2.bar(x, sub["TotalCnt"], color="#7F7F7F", alpha=0.20, label="Count with call (y > 0)")
            ax2.set_ylabel("Count")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            fig.suptitle(
                f"{var}: actual vs predicted E[y | y > 0] ({segment} {sample})", y=1.02
            )
            plt.tight_layout()
            filename = (
                f"{prefix}actual_vs_pred_count_{segment.lower()}_{var}_{sample.lower()}.png"
            )
            fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"[saved] plots/{filename}")


# Step 3: Build validdat. Use original-scale features from Auto_train_imputed
# (so we can re-bin numerics the same way as the EDA) joined with nb_pred for
# the model's predicted E[Y | Y > 0]. Only positive-count rows survive the JOIN
# because nb_pred only contains ids from the count-stage train/val splits
# (which were filtered to call_counts > 0 in the data prep).
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
        JOIN nb_pred AS b
        ON a.id = b.id;
""")

auto_valid_table = build_valid_table(conn, auto_varlist)
plot_actual_vs_pred(auto_valid_table, auto_varlist, "Auto", "13_")

conn.close()
