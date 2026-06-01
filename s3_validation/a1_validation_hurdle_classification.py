import os
import pickle

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from s1_data.a0_setup_directories import auto_model_dir, non_auto_model_dir
from s1_data.db_utils import load_df

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


def build_valid_table(conn, varlist):
    """Aggregate actual and predicted P(y > 0) per (variable, level, sample)."""
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE validTable (
            Variable        VARCHAR,
            ValVar          VARCHAR,
            Sample          VARCHAR,
            PredCnt         DOUBLE,
            ActualCnt       INT,
            TotalCnt        INT,
            SumGivenPositive DOUBLE
        );
    """)
    for var in varlist:
        conn.execute(f"""
            INSERT INTO validTable
                SELECT
                    '{var}' AS Variable,
                    {var}::VARCHAR AS ValVar,
                    Sample,
                    SUM(pred) AS PredCnt,
                    SUM(IF(call_counts > 0, 1, 0)) AS ActualCnt,
                    COUNT(*) AS TotalCnt,
                    SUM(IF(call_counts > 0, call_counts, 0)) AS SumGivenPositive
                FROM validdat
                GROUP BY 1, 2, 3
                ORDER BY 1, 2, 3
        """)
    valid_table = conn.execute(
        "SELECT * FROM validTable ORDER BY Variable, Sample, ValVar;"
    ).fetch_df()
    valid_table["ActualRate"] = valid_table["ActualCnt"] / valid_table["TotalCnt"]
    valid_table["PredRate"] = valid_table["PredCnt"] / valid_table["TotalCnt"]
    valid_table["MeanGivenPositive"] = np.where(
        valid_table["ActualCnt"] > 0,
        valid_table["SumGivenPositive"] / valid_table["ActualCnt"],
        np.nan,
    )
    return valid_table


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
            ax1.plot(x, sub["ActualRate"], color="#C44E52", marker="o", linewidth=2,
                     label="Actual P(y > 0)")
            ax1.plot(x, sub["PredRate"], color="#4C72B0", marker="s", linewidth=2,
                     linestyle="--", label="Predicted P(y > 0)")
            ax1.set_xlabel(var)
            ax1.set_ylabel("P(call_counts > 0)")
            max_rate = float(max(sub["ActualRate"].max(), sub["PredRate"].max()))
            ax1.set_ylim(0, min(1.0, max_rate * 1.15 + 0.05))
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, rotation=45, ha="right")

            ax2 = ax1.twinx()
            ax2.bar(x, sub["TotalCnt"], color="#7F7F7F", alpha=0.20, label="Total count")
            ax2.set_ylabel("Count")

            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

            fig.suptitle(
                f"{var}: actual vs predicted P(y > 0) ({segment} {sample})", y=1.02
            )
            plt.tight_layout()
            filename = (
                f"{prefix}actual_vs_pred_{segment.lower()}_{var}_{sample.lower()}.png"
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
        JOIN lr_pred AS b
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
        JOIN lr_pred AS b
        ON a.id = b.id;
"""

SEGMENTS = [
    ("auto",    auto_model_dir,     AUTO_VARLIST,    AUTO_VALIDDAT_SQL,    "Auto",    "12_"),
    ("nonauto", non_auto_model_dir, NONAUTO_VARLIST, NONAUTO_VALIDDAT_SQL, "NonAuto", "15_"),
]


for segment, model_dir, varlist, validdat_sql, segment_label, prefix in SEGMENTS:
    print("=" * 80)
    print(f"Hurdle stage 1 calibration ({segment_label})")
    print("=" * 80)

    conn = duckdb.connect(database=database_path, read_only=True)

    lr_model = load_pkl(os.path.join(model_dir, f"lr_model_{segment}.pkl"))

    # Keep `id` so we can join predictions back to original-scale features.
    X_train = load_df(conn, f"X_train_{segment}_binary")
    X_val = load_df(conn, f"X_val_{segment}_binary")

    X_train_const = sm.add_constant(X_train.drop(columns=["id"]), has_constant="add")
    X_val_const = sm.add_constant(X_val.drop(columns=["id"]), has_constant="add")

    lr_train_pred = lr_model.predict(X_train_const)
    lr_val_pred = lr_model.predict(X_val_const)

    lr_pred = pd.concat(
        [
            pd.DataFrame({"id": X_train["id"].to_numpy(), "pred": lr_train_pred.to_numpy(), "Sample": "Train"}),
            pd.DataFrame({"id": X_val["id"].to_numpy(),   "pred": lr_val_pred.to_numpy(),   "Sample": "Val"}),
        ],
        axis=0,
        ignore_index=True,
    )
    conn.register("lr_pred", lr_pred)

    conn.execute(validdat_sql)
    valid_table = build_valid_table(conn, varlist)
    plot_actual_vs_pred(valid_table, varlist, segment_label, prefix)

    conn.close()
