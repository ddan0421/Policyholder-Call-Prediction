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

conn = duckdb.connect(database=database_path, read_only=False)


nonauto_train = load_df(conn, "NonAuto_train_imputed")
nonauto_test = load_df(conn, "NonAuto_test_imputed")

auto_train = load_df(conn, "Auto_train_imputed")
auto_test = load_df(conn, "Auto_train_imputed")


# =====================================================================
# Section 1: Call counts analysis (response variable)
#   01_ overall histogram + summary stats per segment
#   02_ non-zero-only histogram + summary stats per segment (stage-2 view)
# =====================================================================

# Call counts analysis: summary stats + histogram per segment
for name, df in [("Auto", auto_train), ("NonAuto", nonauto_train)]:
    s = df["call_counts"]
    print(f"=== {name} ===")
    print(s.describe())
    print(f"mean: {s.mean():.4f}")
    print(f"var:  {s.var():.4f}")
    print(f"P(call_counts == 0): {(s == 0).mean():.4f}")
    print()

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

axes[0].hist(auto_train["call_counts"], bins=30, color="blue", edgecolor="white")
axes[0].set_title("Auto - Histogram of Call Counts")
axes[0].set_xlabel("Number of Calls")
axes[0].set_ylabel("Frequency")

axes[1].hist(nonauto_train["call_counts"], bins=30, color="blue", edgecolor="white")
axes[1].set_title("NonAuto - Histogram of Call Counts")
axes[1].set_xlabel("Number of Calls")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
fig.savefig("plots/01_call_counts_histogram.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[saved] plots/01_call_counts_histogram.png")


# Call counts analysis on non-zero rows only (stage-2 view for hurdle)
for name, df in [("Auto", auto_train), ("NonAuto", nonauto_train)]:
    count_data = df.loc[df["call_counts"] != 0, "call_counts"]
    print(f"=== {name} (non-zero) ===")
    print(count_data.describe())
    print(f"mean: {count_data.mean():.4f}")
    print(f"var:  {count_data.var():.4f}")
    print()

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

axes[0].hist(
    auto_train.loc[auto_train["call_counts"] != 0, "call_counts"],
    bins=30, color="blue", edgecolor="white",
)
axes[0].set_title("Auto - Histogram of Non-zero Call Counts")
axes[0].set_xlabel("Number of Calls")
axes[0].set_ylabel("Frequency")

axes[1].hist(
    nonauto_train.loc[nonauto_train["call_counts"] != 0, "call_counts"],
    bins=30, color="blue", edgecolor="white",
)
axes[1].set_title("NonAuto - Histogram of Non-zero Call Counts")
axes[1].set_xlabel("Number of Calls")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
fig.savefig("plots/02_call_counts_histogram_nonzero.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[saved] plots/02_call_counts_histogram_nonzero.png")

"""
Notes on call_counts modeling

call_counts is overdispersed (variance >> mean), which violates the Poisson
assumption that mean == variance. It also has many zeros, motivating either
a zero-inflated or hurdle approach.

Reference:
https://stats.stackexchange.com/questions/81457/what-is-the-difference-between-zero-inflated-and-hurdle-models

Both models are described in two parts:
1. On/off part (binary): system is "off" with probability pi (only zeros) and
   "on" with probability 1 - pi. Same in both models.
2. Counting part (when "on"): this is where the two differ.
     - Zero-inflated: count distribution can still produce zeros.
     - Hurdle:        count distribution is zero-truncated (must be > 0).

Equivalently:
- Hurdle assumes one process generates zeros (the off state).
- Zero-inflated assumes two processes can generate zeros (off state OR the
  count distribution producing a zero).

Plan in this repo (see README):
- Baselines:    ZIP, ZINB.
- Hurdle:       stage 1 logistic for P(Y > 0); stage 2 zero-truncated NB for
                E[Y | Y > 0] on rows with Y > 0.
                Combined prediction: y_hat = P(Y > 0) * E[Y | Y > 0].
- Experiment order: ZIP -> ZINB -> Hurdle(binary + zero-truncated NB).
"""
# =====================================================================
# Section 2: Auto vs NonAuto variable comparison (03_)
#   - box_whisker_col -> vertical box-and-whisker
#   - bar_col         -> bar chart
#   - if a variable exists in only one segment, show a single plot
# =====================================================================

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
        filename = f"03_box_whisker_{col_slug}_auto_vs_nonauto.png"
        fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] plots/{filename}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    if auto_s is not None:
        sns.boxplot(y=auto_s, ax=ax, color="#4C72B0")
        ax.set_title(f"Auto only - {col}")
        filename = f"03_box_whisker_{col_slug}_auto_only.png"
    else:
        sns.boxplot(y=nonauto_s, ax=ax, color="#55A868")
        ax.set_title(f"NonAuto only - {col}")
        filename = f"03_box_whisker_{col_slug}_nonauto_only.png"

    ax.set_xlabel("")
    ax.set_ylabel(col)
    plt.tight_layout()
    fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] plots/{filename}")


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
        filename = f"03_bar_{col_slug}_auto_vs_nonauto.png"
        fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] plots/{filename}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    if auto_s is not None:
        counts = auto_s.value_counts(dropna=False)
        _plot_bar_counts(ax, counts, col, "#4C72B0")
        ax.set_title(f"Auto only - {col}")
        filename = f"03_bar_{col_slug}_auto_only.png"
    else:
        counts = nonauto_s.value_counts(dropna=False)
        _plot_bar_counts(ax, counts, col, "#55A868")
        ax.set_title(f"NonAuto only - {col}")
        filename = f"03_bar_{col_slug}_nonauto_only.png"
    plt.tight_layout()
    fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] plots/{filename}")


# Generate all requested comparisons
for col in box_whisker_col:
    compare_box_whisker_variable(auto_train, nonauto_train, col)

for col in bar_col:
    compare_bar_variable(auto_train, nonauto_train, col)



# =====================================================================
# Section 3: Cap household_policy_counts (04_)
#   uncapped vs capped, side-by-side bar charts (Auto + NonAuto)
# =====================================================================

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
filename = f"04_bar_{col.replace(' ', '_')}_raw_vs_capped.png"
fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[saved] plots/{filename}")


# =====================================================================
# Section 4: Call rate, marginal mean, and conditional mean by predictor level
#   05_ Auto    call_rate           (P(y > 0))
#   06_ Auto    marginal_mean       (E[y], all rows)
#   07_ Auto    conditional_mean    (E[y | y > 0])
#   08_ NonAuto call_rate           (P(y > 0))
#   09_ NonAuto marginal_mean       (E[y], all rows)
#   10_ NonAuto conditional_mean    (E[y | y > 0])
# =====================================================================

def _sort_by_valvar(df):
    try:
        out = df.copy()
        out["_sort_key"] = out["ValVar"].astype(float)
        return out.sort_values("_sort_key").drop(columns="_sort_key")
    except ValueError:
        return df.sort_values("ValVar")


def build_valid_table(conn, varlist):
    """Aggregate stage-1 (P(y>0)) and stage-2 (E[y|y>0]) stats per variable level."""
    conn.execute("""
        CREATE OR REPLACE TEMP TABLE validTable (
            Variable        VARCHAR,
            ValVar          VARCHAR,
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
                    SUM(IF(call_counts > 0, 1, 0)) AS ActualCnt,
                    COUNT(*) AS TotalCnt,
                    SUM(IF(call_counts > 0, call_counts, 0)) AS SumGivenPositive
                FROM validdat
                GROUP BY 1, 2
                ORDER BY 1, 2
        """)
    valid_table = conn.execute(
        "SELECT * FROM validTable ORDER BY Variable, ValVar;"
    ).fetch_df()
    valid_table["Rate"] = valid_table["ActualCnt"] / valid_table["TotalCnt"]
    valid_table["MeanGivenPositive"] = np.where(
        valid_table["ActualCnt"] > 0,
        valid_table["SumGivenPositive"] / valid_table["ActualCnt"],
        np.nan,
    )
    # Marginal E[Y] per bin: average of call_counts over ALL rows (zeros included).
    # SumGivenPositive == SUM(call_counts) since rows with call_counts == 0
    # contribute 0 to the IF-sum either way.
    valid_table["MarginalMean"] = valid_table["SumGivenPositive"] / valid_table["TotalCnt"]
    return valid_table


def plot_call_rate_curves(valid_table, varlist, segment, prefix):
    """One PNG per variable: P(y > 0) line on left axis, counts as bars on right."""
    for var in varlist:
        sub = _sort_by_valvar(valid_table[valid_table["Variable"] == var])
        if sub.empty:
            print(f"[skip] {segment} {var}: no rows in validTable")
            continue

        x = np.arange(len(sub))
        labels = sub["ValVar"].astype(str)

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(x, sub["Rate"], color="#C44E52", marker="o", linewidth=2, label="P(y > 0)")
        ax1.set_xlabel(var)
        ax1.set_ylabel("P(call_counts > 0)")
        ax1.set_ylim(0, min(1.0, sub["Rate"].max() * 1.15 + 0.05))
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right")

        ax2 = ax1.twinx()
        ax2.bar(x, sub["TotalCnt"], color="#4C72B0", alpha=0.65, label="Total count")
        ax2.bar(x, sub["ActualCnt"], color="#55A868", alpha=0.85, label="Count with call (y > 0)")
        ax2.set_ylabel("Count")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        fig.suptitle(f"{var}: call rate by level ({segment} train)", y=1.02)
        plt.tight_layout()
        filename = f"{prefix}call_rate_{segment.lower()}_{var}.png"
        fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] plots/{filename}")


def plot_marginal_mean_curves(valid_table, varlist, segment, prefix):
    """Marginal view: E[Y] (average call_counts including zeros) per bin."""
    for var in varlist:
        sub = _sort_by_valvar(valid_table[valid_table["Variable"] == var])
        if sub.empty:
            print(f"[skip] {segment} {var}: no rows in validTable")
            continue

        x = np.arange(len(sub))
        labels = sub["ValVar"].astype(str)

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(x, sub["MarginalMean"], color="#C44E52", marker="o", linewidth=2,
                 label="E[y]")
        ax1.set_xlabel(var)
        ax1.set_ylabel("Marginal mean call_counts")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right")

        ax2 = ax1.twinx()
        ax2.bar(x, sub["TotalCnt"], color="#4C72B0", alpha=0.4, label="Total count")
        ax2.set_ylabel("Count")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        fig.suptitle(f"{var}: marginal mean call_counts by level ({segment} train)", y=1.02)
        plt.tight_layout()
        filename = f"{prefix}marginal_mean_{segment.lower()}_{var}.png"
        fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] plots/{filename}")


def plot_conditional_mean_curves(valid_table, varlist, segment, prefix):
    """Stage-2 view: E[call_counts | call_counts > 0] per bin, with positive-count bars."""
    for var in varlist:
        sub = _sort_by_valvar(valid_table[valid_table["Variable"] == var])
        if sub.empty:
            print(f"[skip] {segment} {var}: no rows in validTable")
            continue

        x = np.arange(len(sub))
        labels = sub["ValVar"].astype(str)

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(x, sub["MeanGivenPositive"], color="#C44E52", marker="o", linewidth=2,
                 label="E[y | y > 0]")
        ax1.set_xlabel(var)
        ax1.set_ylabel("E[call_counts | call_counts > 0]")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha="right")

        ax2 = ax1.twinx()
        ax2.bar(x, sub["ActualCnt"], color="#55A868", alpha=0.65, label="Count with call (y > 0)")
        ax2.set_ylabel("Count")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        fig.suptitle(f"{var}: conditional mean call_counts given y > 0 ({segment} train)", y=1.02)
        plt.tight_layout()
        filename = f"{prefix}conditional_mean_{segment.lower()}_{var}.png"
        fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] plots/{filename}")


# Auto: build validdat with Auto-specific bins/caps, then plot call rate / mean curves
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
            LEAST(("12m_call_history" / 5)::INT * 5, 40) AS binned_12m_call_history,
            acq_method,
            LEAST((ann_prm_amt / 900)::INT * 900, 7200) AS binned_ann_prm_amt,
            bi_limit_group,
            digital_contact_ind,
            geo_group,
            has_prior_carrier,
            (home_lot_sq_footage / 10000)::INT * 10000 AS binned_home_lot_sq_footage,
            household_group,
            LEAST(household_policy_counts, 4) AS capped_household_policy_counts,
            newest_veh_age,
            pay_type_code,
            pol_edeliv_ind_filled,
            prdct_sbtyp_grp,
            product_sbtyp,
            telematics_ind,
            LEAST((tenure_at_snapshot / 100)::INT * 100, 500) AS binned_tenure_at_snapshot,
            call_counts
        FROM Auto_train_imputed;
""")
auto_valid_table = build_valid_table(conn, auto_varlist)
plot_call_rate_curves(auto_valid_table, auto_varlist, "Auto", "05_")
plot_marginal_mean_curves(auto_valid_table, auto_varlist, "Auto", "06_")
plot_conditional_mean_curves(auto_valid_table, auto_varlist, "Auto", "07_")



# NonAuto: build validdat with NonAuto-specific bins/caps, then plot call rate / mean curves
nonauto_varlist = [
    "binned_12m_call_history", "acq_method", "binned_ann_prm_amt",
    "digital_contact_ind", "geo_group", "has_prior_carrier", "binned_home_lot_sq_footage",
    "household_group", "capped_household_policy_counts",
    "pay_type_code", "pol_edeliv_ind_filled", "prdct_sbtyp_grp", "product_sbtyp",
    "trm_len_mo", "binned_tenure_at_snapshot",
]

conn.execute("""
    CREATE OR REPLACE TEMP TABLE validdat AS
        SELECT
            LEAST(("12m_call_history" / 5)::INT * 5, 30) AS binned_12m_call_history,
            acq_method,
            LEAST((ann_prm_amt / 900)::INT * 900, 7200) AS binned_ann_prm_amt,
            digital_contact_ind,
            geo_group,
            has_prior_carrier,
            (home_lot_sq_footage / 10000)::INT * 10000 AS binned_home_lot_sq_footage,
            household_group,
            LEAST(household_policy_counts, 5) AS capped_household_policy_counts,
            pay_type_code,
            pol_edeliv_ind_filled,
            prdct_sbtyp_grp,
            product_sbtyp,
            LEAST((tenure_at_snapshot / 100)::INT * 100, 500) AS binned_tenure_at_snapshot,
            trm_len_mo,
            call_counts
        FROM NonAuto_train_imputed;
""")
nonauto_valid_table = build_valid_table(conn, nonauto_varlist)
plot_call_rate_curves(nonauto_valid_table, nonauto_varlist, "NonAuto", "08_")
plot_marginal_mean_curves(nonauto_valid_table, nonauto_varlist, "NonAuto", "09_")
plot_conditional_mean_curves(nonauto_valid_table, nonauto_varlist, "NonAuto", "10_")



# =====================================================================
# Section 5: Cap + log1p transform comparison (11_)
#   raw vs log1p(min(x, cap)) using vertical boxplots, Auto vs NonAuto
#   Caps match the binning thresholds applied per segment above.
# =====================================================================

# Cap + log1p comparison (raw vs log1p(min(x, cap))) using vertical boxplots
log_caps = {
    "12m_call_history":   {"Auto": 40,   "NonAuto": 30},
    "ann_prm_amt":        {"Auto": 7200, "NonAuto": 7200},
    "tenure_at_snapshot": {"Auto": 500,  "NonAuto": 500},
}

for col, caps in log_caps.items():
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
            pd.DataFrame({"segment": "Auto", "value": np.log1p(np.minimum(auto_s, caps["Auto"]))})
        )
    if nonauto_s is not None:
        trans_parts.append(
            pd.DataFrame({"segment": "NonAuto", "value": np.log1p(np.minimum(nonauto_s, caps["NonAuto"]))})
        )
    trans_df = pd.concat(trans_parts, ignore_index=True)
    sns.boxplot(data=trans_df, x="segment", y="value", ax=axes[1])
    axes[1].set_title(f"{col} (log1p with cap: Auto {caps['Auto']}, NonAuto {caps['NonAuto']})")
    axes[1].set_xlabel("")
    axes[1].set_ylabel(f"log1p(min({col}, cap))")

    plt.tight_layout()
    filename = f"11_box_whisker_{col.replace(' ', '_')}_raw_vs_capped_log1p.png"
    fig.savefig(f"plots/{filename}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] plots/{filename}")


conn.close()