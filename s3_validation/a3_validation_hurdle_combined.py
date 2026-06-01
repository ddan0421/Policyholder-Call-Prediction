import os
import pickle
import warnings

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

from s1_data.a0_setup_directories import auto_model_dir, non_auto_model_dir
from s1_data.db_utils import load_df
from s1_data.transform_utils import apply_transformer, load_transformer

warnings.filterwarnings("ignore")

"""
Combined hurdle calibration for Auto and NonAuto segments.

y_hat(X) = P(Y > 0 | X) * E[Y | Y > 0, X]

Aggregates by predictor level so we can plot actual vs predicted marginal
mean E[Y] (zeros included), mirroring the EDA `plot_marginal_mean_curves`
view (06_/09_marginal_mean_*).

Saved as 14_ (Auto) and 17_ (NonAuto) actual_vs_pred_marginal_*.
"""

base_folder = "data"
database = "TravelersPolicyHolderCall.duckdb"
database_path = os.path.join(base_folder, database)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _sort_by_valvar(df):
    try:
        out = df.copy()
        out["_sort_key"] = out["ValVar"].astype(float)
        return out.sort_values("_sort_key").drop(columns="_sort_key")
    except ValueError:
        return df.sort_values("ValVar")


def transform_and_constant(df_raw, transformer):
    X = apply_transformer(df_raw, transformer)
    X = X.drop(columns=["id"])
    return sm.add_constant(X, has_constant="add")


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


AUTO_VARLIST = [
    "binned_12m_call_history", "acq_method", "binned_ann_prm_amt", "bi_limit_group",
    "digital_contact_ind", "geo_group", "has_prior_carrier", "binned_home_lot_sq_footage",
    "household_group", "capped_household_policy_counts", "newest_veh_age",
    "pay_type_code", "pol_edeliv_ind_filled", "prdct_sbtyp_grp", "product_sbtyp",
    "telematics_ind", "binned_tenure_at_snapshot",
]

NONAUTO_VARLIST = [
    "binned_12m_call_history", "acq_method", "binned_ann_prm_amt",
    "digital_contact_ind", "geo_group", "has_prior_carrier", "binned_home_lot_sq_footage",
    "household_group", "capped_household_policy_counts",
    "pay_type_code", "pol_edeliv_ind_filled", "prdct_sbtyp_grp", "product_sbtyp",
    "trm_len_mo", "binned_tenure_at_snapshot",
]

AUTO_VALIDDAT_SQL = """
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
"""

NONAUTO_VALIDDAT_SQL = """
    CREATE OR REPLACE TEMP TABLE validdat AS
        SELECT
            LEAST((a."12m_call_history" / 5)::INT * 5, 30) AS binned_12m_call_history,
            a.acq_method,
            LEAST((a.ann_prm_amt / 900)::INT * 900, 7200) AS binned_ann_prm_amt,
            a.digital_contact_ind,
            a.geo_group,
            a.has_prior_carrier,
            (a.home_lot_sq_footage / 10000)::INT * 10000 AS binned_home_lot_sq_footage,
            a.household_group,
            LEAST(a.household_policy_counts, 5) AS capped_household_policy_counts,
            a.pay_type_code,
            a.pol_edeliv_ind_filled,
            a.prdct_sbtyp_grp,
            a.product_sbtyp,
            LEAST((a.tenure_at_snapshot / 100)::INT * 100, 500) AS binned_tenure_at_snapshot,
            a.trm_len_mo,
            a.call_counts,
            b.Sample,
            b.pred
        FROM NonAuto_train_imputed AS a
        JOIN hurdle_pred AS b
        ON a.id = b.id;
"""

SEGMENTS = [
    ("auto",    auto_model_dir,     "Auto",    AUTO_VARLIST,    AUTO_VALIDDAT_SQL,    "14_"),
    ("nonauto", non_auto_model_dir, "NonAuto", NONAUTO_VARLIST, NONAUTO_VALIDDAT_SQL, "17_"),
]


for segment, model_dir, segment_label, varlist, validdat_sql, prefix in SEGMENTS:
    print("=" * 80)
    print(f"Hurdle combined calibration ({segment_label})")
    print("=" * 80)

    conn = duckdb.connect(database=database_path, read_only=True)

    lr_model = load_pkl(os.path.join(model_dir, f"lr_model_{segment}.pkl"))
    nb_model = load_pkl(os.path.join(model_dir, f"nb_model_{segment}.pkl"))
    binary_transformer = load_transformer(
        os.path.join(model_dir, f"{segment}_binary_transformer.pkl")
    )
    count_transformer = load_transformer(
        os.path.join(model_dir, f"{segment}_count_nb_transformer.pkl")
    )

    # Recover Train/Val rows by re-running the same train_test_split used at training time.
    X_raw = load_df(conn, f"{segment_label}_train_binary", exclude_cols=["nonzero_call"])
    y_raw = load_df(conn, f"{segment_label}_train_imputed")[["id", "call_counts"]]

    X_train_raw, X_val_raw, _, _ = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42
    )

    def hurdle_predict(df_raw):
        X_binary = transform_and_constant(df_raw, binary_transformer)
        X_count = transform_and_constant(df_raw, count_transformer)
        prob_y_gt_0 = np.asarray(lr_model.predict(X_binary))
        mu_y_given_pos = np.asarray(nb_model.predict(X_count))
        return prob_y_gt_0 * mu_y_given_pos

    train_pred = hurdle_predict(X_train_raw)
    val_pred = hurdle_predict(X_val_raw)

    hurdle_pred = pd.concat(
        [
            pd.DataFrame({"id": X_train_raw["id"].to_numpy(), "pred": train_pred, "Sample": "Train"}),
            pd.DataFrame({"id": X_val_raw["id"].to_numpy(),   "pred": val_pred,   "Sample": "Val"}),
        ],
        axis=0,
        ignore_index=True,
    )
    conn.register("hurdle_pred", hurdle_pred)

    conn.execute(validdat_sql)
    valid_table = build_valid_table(conn, varlist)
    plot_actual_vs_pred(valid_table, varlist, segment_label, prefix)

    conn.close()
