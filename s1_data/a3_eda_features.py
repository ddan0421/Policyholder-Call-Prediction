import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from s1_data.db_utils import load_df
import duckdb
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

base_folder = "data"
database = "TravelersPolicyHolderCall.duckdb"
database_path = os.path.join(base_folder, database)

conn = duckdb.connect(database=database_path, read_only=True)


nonauto_train = load_df(conn, "NonAuto_train_imputed", add_id=False)
nonauto_test = load_df(conn, "NonAuto_test_imputed", add_id=False)

auto_train = load_df(conn, "Auto_train_imputed", add_id=False)
auto_test = load_df(conn, "Auto_train_imputed", add_id=False)


numeric_col = ["12m_call_history", "ann_prm_amt", "home_lot_sq_footage", "household_policy_counts", "newest_veh_age",
                "tenure_at_snapshot"]

cat_col = ["acq_method", "bi_limit_group", "channel", "digital_contact_ind", "geo_group", "has_prior_carrier",
            "household_group", "pay_type_code", "pol_edeliv_ind_filled", "prdct_sbtyp_grp", "product_sbtyp", "telematics_ind",
            "trm_len_mo"]



# Auto vs NonAuto variable comparison
# - numeric_col: vertical box-and-whisker
# - cat_col: bar chart
# - if a variable exists in only one segment, show a single plot

def _series_if_exists(df, col):
    return df[col] if col in df.columns else None


def compare_numeric_variable(auto_df, nonauto_df, col):
    auto_s = _series_if_exists(auto_df, col)
    nonauto_s = _series_if_exists(nonauto_df, col)
    col_slug = col.replace(" ", "_")

    if auto_s is None and nonauto_s is None:
        print(f"[skip] {col}: not found in Auto or NonAuto")
        return

    if auto_s is not None and nonauto_s is not None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        sns.boxplot(y=auto_s, ax=axes[0], color="#4C72B0")
        axes[0].set_title(f"Auto - {col}")
        axes[0].set_xlabel("")
        axes[0].set_ylabel(col)

        sns.boxplot(y=nonauto_s, ax=axes[1], color="#55A868")
        axes[1].set_title(f"NonAuto - {col}")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("")

        fig.suptitle(f"Numeric comparison: {col}", y=1.02)
        plt.tight_layout()
        filename = f"numeric_{col_slug}_auto_vs_nonauto.png"
        fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
        return

    # only one segment has this variable
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if auto_s is not None:
        sns.boxplot(y=auto_s, ax=ax, color="#4C72B0")
        ax.set_title(f"Auto only - {col}")
    else:
        sns.boxplot(y=nonauto_s, ax=ax, color="#55A868")
        ax.set_title(f"NonAuto only - {col}")

    ax.set_xlabel("")
    ax.set_ylabel(col)
    plt.tight_layout()
    if auto_s is not None:
        filename = f"numeric_{col_slug}_auto_only.png"
    else:
        filename = f"numeric_{col_slug}_nonauto_only.png"
    fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")


def _category_order(auto_s, nonauto_s):
    combined = pd.concat([auto_s, nonauto_s], ignore_index=True)
    return combined.value_counts(dropna=False).index.tolist()


def compare_categorical_variable(auto_df, nonauto_df, col):
    auto_s = _series_if_exists(auto_df, col)
    nonauto_s = _series_if_exists(nonauto_df, col)
    col_slug = col.replace(" ", "_")

    if auto_s is None and nonauto_s is None:
        print(f"[skip] {col}: not found in Auto or NonAuto")
        return

    if auto_s is not None:
        auto_s = auto_s.astype("string").fillna("<NA>")
    if nonauto_s is not None:
        nonauto_s = nonauto_s.astype("string").fillna("<NA>")

    if auto_s is not None and nonauto_s is not None:
        order = _category_order(auto_s, nonauto_s)

        auto_counts = auto_s.value_counts(dropna=False).reindex(order, fill_value=0)
        nonauto_counts = nonauto_s.value_counts(dropna=False).reindex(order, fill_value=0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

        axes[0].bar(auto_counts.index, auto_counts.values, color="#4C72B0")
        axes[0].set_title(f"Auto - {col}")
        axes[0].set_xlabel(col)
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=45)

        axes[1].bar(nonauto_counts.index, nonauto_counts.values, color="#55A868")
        axes[1].set_title(f"NonAuto - {col}")
        axes[1].set_xlabel(col)
        axes[1].set_ylabel("")
        axes[1].tick_params(axis="x", rotation=45)

        fig.suptitle(f"Categorical comparison: {col}", y=1.02)
        plt.tight_layout()
        filename = f"categorical_{col_slug}_auto_vs_nonauto.png"
        fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
        return

    # only one segment has this variable
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    if auto_s is not None:
        counts = auto_s.value_counts(dropna=False)
        ax.bar(counts.index, counts.values, color="#4C72B0")
        ax.set_title(f"Auto only - {col}")
    else:
        counts = nonauto_s.value_counts(dropna=False)
        ax.bar(counts.index, counts.values, color="#55A868")
        ax.set_title(f"NonAuto only - {col}")

    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    if auto_s is not None:
        filename = f"categorical_{col_slug}_auto_only.png"
    else:
        filename = f"categorical_{col_slug}_nonauto_only.png"
    fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")


# Generate all requested comparisons
for col in numeric_col:
    compare_numeric_variable(auto_train, nonauto_train, col)

for col in cat_col:
    compare_categorical_variable(auto_train, nonauto_train, col)



# Log-transform comparison (before vs after) using vertical boxplots
# no need to transform "home_lot_sq_footage" "newest_veh_age" since their distribution is already normal
# household_policy_counts needs to get capped or binned
log_col = ["12m_call_history", "ann_prm_amt", "tenure_at_snapshot"]

for col in log_col:
    auto_s = auto_train[col] if col in auto_train.columns else None
    nonauto_s = nonauto_train[col] if col in nonauto_train.columns else None

    if auto_s is None and nonauto_s is None:
        print(f"[skip] {col}: not found in Auto or NonAuto")
        continue

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: raw
    raw_parts = []
    if auto_s is not None:
        raw_parts.append(pd.DataFrame({"segment": "Auto", "value": auto_s}))
    if nonauto_s is not None:
        raw_parts.append(pd.DataFrame({"segment": "NonAuto", "value": nonauto_s}))
    raw_df = pd.concat(raw_parts, ignore_index=True)
    sns.boxplot(data=raw_df, x="segment", y="value", ax=axes[0])
    axes[0].set_title(f"{col} (Raw)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel(col)

    # Right: transformed
    trans_parts = []
    if auto_s is not None:
        trans_parts.append(
            pd.DataFrame({"segment": "Auto", "value": np.log1p(auto_s)})
        )
    if nonauto_s is not None:
        trans_parts.append(
            pd.DataFrame({"segment": "NonAuto", "value": np.log1p(nonauto_s)})
        )
    trans_df = pd.concat(trans_parts, ignore_index=True)
    sns.boxplot(data=trans_df, x="segment", y="value", ax=axes[1])
    axes[1].set_title(f"{col} (log1p)")
    axes[1].set_xlabel("")
    axes[1].set_ylabel(f"log1p({col})")

    plt.tight_layout()
    filename = f"numeric_{col.replace(' ', '_')}_raw_vs_log1p.png"
    fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
    plt.close(fig)

