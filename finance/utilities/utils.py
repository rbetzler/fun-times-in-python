import os
import psycopg2
import pandas as pd
import datetime


# DW_STOCKS = 'postgresql://postgres:password@172.18.0.2:5432/dw_stocks'
DW_STOCKS = 'postgresql://postgres:password@172.17.0.6:5432/dw_stocks'
DW_STOCKS_JUPYTER = 'postgresql://postgres:password@localhost:5432/dw_stocks'


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


def retrieve_secret(var, pwd=''):
    secrets = open('audit/babylon.env').read().split()
    for secret in secrets:
        if var == secret.split('=')[0]:
            pwd = secret.split('=')[1]
    return pwd


def create_directory(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def iter_date_range(start_date, end_date):
    if not isinstance(start_date, datetime.datetime):
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    if not isinstance(end_date, datetime.datetime):
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    days_diff = (end_date - start_date).days
    for n in range(days_diff):
        yield start_date + datetime.timedelta(n)
