import os
import psycopg2
import pandas as pd
import datetime

_SECRETS = 'audit/babylon.env'
_SECRETS_JUPYTER = '/usr/src/app/audit/babylon.env'  # stupid jupyter workaround


# Secrets
def retrieve_secret(var, pwd=''):
    file = _SECRETS if os.path.isfile(_SECRETS) else _SECRETS_JUPYTER
    secrets = open(file).read().split()
    for secret in secrets:
        _var, _, _pwd = secret.partition('=')
        if var == _var:
            pwd = _pwd
    return pwd


# Db utils
DW_STOCKS = retrieve_secret('DW_STOCKS')
DW_STOCKS_JUPYTER = retrieve_secret('DW_STOCKS_JUPYTER')


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


# Other
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
