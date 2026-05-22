import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


box_whisker_col = ["12m_call_history", "ann_prm_amt", "home_lot_sq_footage", "newest_veh_age",
                   "tenure_at_snapshot"]

bar_col = ["household_policy_counts", "acq_method", "bi_limit_group", "channel", "digital_contact_ind",
           "geo_group", "has_prior_carrier", "household_group", "pay_type_code", "pol_edeliv_ind_filled",
           "prdct_sbtyp_grp", "product_sbtyp", "telematics_ind", "trm_len_mo"]


# Auto vs NonAuto variable comparison
# - box_whisker_col: vertical box-and-whisker
# - bar_col: bar chart
# - if a variable exists in only one segment, show a single plot

def _series_if_exists(df, col):
    return df[col] if col in df.columns else None


def compare_box_whisker_variable(auto_df, nonauto_df, col):
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

        fig.suptitle(f"Box-and-whisker comparison: {col}", y=1.02)
        plt.tight_layout()
        filename = f"box_whisker_{col_slug}_auto_vs_nonauto.png"
        fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if auto_s is not None:
        sns.boxplot(y=auto_s, ax=ax, color="#4C72B0")
        ax.set_title(f"Auto only - {col}")
        filename = f"box_whisker_{col_slug}_auto_only.png"
    else:
        sns.boxplot(y=nonauto_s, ax=ax, color="#55A868")
        ax.set_title(f"NonAuto only - {col}")
        filename = f"box_whisker_{col_slug}_nonauto_only.png"

    ax.set_xlabel("")
    ax.set_ylabel(col)
    plt.tight_layout()
    fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _as_bar_categories(series):
    """Bar charts always use categorical x-axis labels, not continuous numeric scale."""
    return series.astype("string").fillna("<NA>")


def _bar_value_order(auto_s, nonauto_s, col):
    combined = pd.concat([auto_s, nonauto_s], ignore_index=True)
    order = combined.value_counts(dropna=False).index.tolist()
    if col == "household_policy_counts":
        return sorted(order, key=lambda x: float(x))
    return order


def _plot_bar_counts(ax, counts, col, color):
    labels = [str(x) for x in counts.index]
    x_pos = np.arange(len(labels))
    ax.bar(x_pos, counts.values, color=color)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")


def _draw_bar_panel(ax, series, col, segment_label, color):
    counts = _as_bar_categories(series).value_counts(dropna=False)
    if col == "household_policy_counts":
        counts = counts.reindex(sorted(counts.index, key=float))
    else:
        counts = counts.sort_index()
    _plot_bar_counts(ax, counts, col, color)
    ax.set_title(f"{segment_label} - {col}")


def compare_bar_variable(auto_df, nonauto_df, col):
    auto_s = _series_if_exists(auto_df, col)
    nonauto_s = _series_if_exists(nonauto_df, col)
    col_slug = col.replace(" ", "_")

    if auto_s is None and nonauto_s is None:
        print(f"[skip] {col}: not found in Auto or NonAuto")
        return

    if auto_s is not None:
        auto_s = _as_bar_categories(auto_s)
    if nonauto_s is not None:
        nonauto_s = _as_bar_categories(nonauto_s)

    if auto_s is not None and nonauto_s is not None:
        order = _bar_value_order(auto_s, nonauto_s, col)

        auto_counts = auto_s.value_counts(dropna=False).reindex(order, fill_value=0)
        nonauto_counts = nonauto_s.value_counts(dropna=False).reindex(order, fill_value=0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

        _plot_bar_counts(axes[0], auto_counts, col, "#4C72B0")
        axes[0].set_title(f"Auto - {col}")

        _plot_bar_counts(axes[1], nonauto_counts, col, "#55A868")
        axes[1].set_title(f"NonAuto - {col}")
        axes[1].set_ylabel("")

        fig.suptitle(f"Bar chart comparison: {col}", y=1.02)
        plt.tight_layout()
        filename = f"bar_{col_slug}_auto_vs_nonauto.png"
        fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    if auto_s is not None:
        counts = auto_s.value_counts(dropna=False)
        _plot_bar_counts(ax, counts, col, "#4C72B0")
        ax.set_title(f"Auto only - {col}")
        filename = f"bar_{col_slug}_auto_only.png"
    else:
        counts = nonauto_s.value_counts(dropna=False)
        _plot_bar_counts(ax, counts, col, "#55A868")
        ax.set_title(f"NonAuto only - {col}")
        filename = f"bar_{col_slug}_nonauto_only.png"
    plt.tight_layout()
    fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
    plt.close(fig)


# Generate all requested comparisons
for col in box_whisker_col:
    compare_box_whisker_variable(auto_train, nonauto_train, col)

for col in bar_col:
    compare_bar_variable(auto_train, nonauto_train, col)


# Log-transform comparison (before vs after) using vertical boxplots
log_col = ["12m_call_history", "ann_prm_amt", "tenure_at_snapshot"]

for col in log_col:
    auto_s = auto_train[col] if col in auto_train.columns else None
    nonauto_s = nonauto_train[col] if col in nonauto_train.columns else None

    if auto_s is None and nonauto_s is None:
        print(f"[skip] {col}: not found in Auto or NonAuto")
        continue

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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
    filename = f"box_whisker_{col.replace(' ', '_')}_raw_vs_log1p.png"
    fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
    plt.close(fig)


# Cap household_policy_counts: uncapped vs capped (bar charts)
col = "household_policy_counts"
auto_cap = 4
nonauto_cap = 5

auto_s = auto_train[col]
nonauto_s = nonauto_train[col]
auto_capped = np.minimum(auto_s, auto_cap)
nonauto_capped = np.minimum(nonauto_s, nonauto_cap)

fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey="row")

_draw_bar_panel(axes[0, 0], auto_s, col, "Auto", "#4C72B0")
_draw_bar_panel(axes[0, 1], nonauto_s, col, "NonAuto", "#55A868")
_draw_bar_panel(axes[1, 0], auto_capped, f"{col} (cap {auto_cap}+)", "Auto", "#4C72B0")
_draw_bar_panel(axes[1, 1], nonauto_capped, f"{col} (cap {nonauto_cap}+)", "NonAuto", "#55A868")

fig.suptitle(f"{col}: uncapped (top) vs capped (bottom)", y=1.02)
plt.tight_layout()
filename = f"bar_{col.replace(' ', '_')}_raw_vs_capped.png"
fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
plt.close(fig)
