def save_df(conn, df, table_name, add_id=True):
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

def load_df(conn, table_name, delete_id=True):
    exclude_id = "EXCLUDE (Id)" if delete_id else ""
    query = f"""
        SELECT * {exclude_id}
        FROM {table_name}
        ORDER BY Id;
    """

    return conn.execute(query).fetch_df()

