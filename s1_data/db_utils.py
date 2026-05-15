def save_df(conn, df, table_name):
    temp_name = f"tmp_{table_name}"
    
    # create copy and add Id
    df_copy = df.copy()
    df_copy["Id"] = range(len(df_copy))
    
    # register + create table
    conn.register(temp_name, df_copy)
    
    conn.execute(f"""
        CREATE OR REPLACE TABLE {table_name} AS 
        SELECT * FROM {temp_name};
    """)
    
    conn.unregister(temp_name)

def load_df(conn, table_name):
    query = f"""
        SELECT * EXCLUDE (Id)
        FROM {table_name}
        ORDER BY Id;
    """
    
    return conn.execute(query).fetch_df()

