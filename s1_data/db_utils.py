import duckdb
import pandas as pd


def save_df(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table_name: str,
    add_id: bool = True,
) -> None:
    temp_name = f"tmp_{table_name}"

    df_copy = df.copy()
    if add_id:
        df_copy["Id"] = range(len(df_copy))

    conn.register(temp_name, df_copy)

    conn.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS 
        SELECT * FROM {temp_name};
    """)

    conn.unregister(temp_name)


def load_df(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    exclude_clause = f"EXCLUDE ({', '.join(exclude_cols)})" if exclude_cols else ""

    query = f"""
        SELECT * {exclude_clause}
        FROM {table_name}
        ORDER BY Id;
    """

    return conn.execute(query).fetch_df()

