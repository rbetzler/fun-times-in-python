import psycopg2
import pandas as pd


# DW_STOCKS = 'postgresql://postgres:password@localhost:5432/dw_stocks'
# DW_STOCKS = 'postgresql://postgres:password@172.18.0.2:5432/dw_stocks'
DW_STOCKS = 'postgresql://postgres:password@172.17.0.1:5432/dw_stocks'


def query_db(db_connection=DW_STOCKS, query=None):
    conn = psycopg2.connect(db_connection)
    df = pd.read_sql(query, conn)
    return df


def insert_record(db_connection=DW_STOCKS, query=None):
    conn = psycopg2.connect(db_connection)
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()
