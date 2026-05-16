import duckdb
import os
from s1_data.gdrive_download import download_from_drive

data_dict = {
    "train.csv": "1tDbkww6-LElQu-dc5elOjiLJbX-Z_kkY",
    "test.csv": "1qceP-c2AS-PHvdmiGJ2qjerbIW1vCFyu"
}
base_folder = "data"
database = "TravelersPolicyHolderCall.duckdb"
database_path = os.path.join(base_folder, database)
train_path = os.path.join(base_folder, "train.csv")
test_path = os.path.join(base_folder, "test.csv")

for filename, file_id in data_dict.items():
    download_from_drive(file_id, filename, base_folder)

conn = duckdb.connect(database = database_path, read_only = False)

# Standard CSV null tokens; categorical missings (e.g. acq_method "missing") stay as literals.
default_na = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN",
              "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NA",
              "NULL", "NaN", "n/a", "na", "nan", "null"]
nullstr_sql = "[" + ", ".join(f"'{x}'" for x in default_na) + "]"

for table, path in {"train": train_path, "test": test_path}.items():
    target_col = ', CAST(call_counts AS INTEGER) AS "call_counts"' if table == "train" else ""
    conn.execute(f"""
        CREATE OR REPLACE TABLE {table} AS
        SELECT
            CAST(id AS INTEGER) AS "id"
            , CAST("12m_call_history" AS INTEGER) AS "12m_call_history"
            , CAST(acq_method AS VARCHAR) AS "acq_method"
            , CAST(ann_prm_amt AS DOUBLE) AS "ann_prm_amt"
            , CAST(bi_limit_group AS VARCHAR) AS "bi_limit_group"
            , CAST(channel AS VARCHAR) AS "channel"
            , CAST(digital_contact_ind AS INTEGER) AS "digital_contact_ind"
            , CAST(geo_group AS VARCHAR) AS "geo_group"
            , CAST(has_prior_carrier AS INTEGER) AS "has_prior_carrier"
            , CAST(home_lot_sq_footage AS DOUBLE) AS "home_lot_sq_footage"
            , CAST(household_group AS VARCHAR) AS "household_group"
            , CAST(household_policy_counts AS INTEGER) AS "household_policy_counts"
            , CAST(newest_veh_age AS INTEGER) AS "newest_veh_age"
            , CAST(pay_type_code AS VARCHAR) AS "pay_type_code"
            , CAST(pol_edeliv_ind AS INTEGER) AS "pol_edeliv_ind"
            , CAST(prdct_sbtyp_grp AS VARCHAR) AS "prdct_sbtyp_grp"
            , CAST(product_sbtyp AS VARCHAR) AS "product_sbtyp"
            , CAST(telematics_ind AS INTEGER) AS "telematics_ind"
            , CAST(tenure_at_snapshot AS INTEGER) AS "tenure_at_snapshot"
            , CAST(trm_len_mo AS INTEGER) AS "trm_len_mo"
            {target_col}
        FROM READ_CSV_AUTO(
            '{path}',
            NULLSTR={nullstr_sql}
        );
                """)

print(conn.execute("SHOW TABLES").fetchall())
conn.close()

print(f"Saved DuckDB database to {database_path}")
