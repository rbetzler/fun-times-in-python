#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:45:16 2019

@author: nautilus
"""

import os
import sys
import psycopg2
import datetime
import pandas as pd
from sqlalchemy import create_engine
from py_scripts.utilities.db_utilities import ConnectionStrings, DbSchemas


#Path to raw file
file_path = '/mnt/dim-stocks/directory-us-tickers.csv'

#Read csv
df = pd.read_csv(file_path, header = None)
df.columns = [
    'ticker',
    'company',
    'adr_tso',
    'ipo_year',
    'sector',
    'industry',
    'exchange'
    ]

#Add timestamp
process_timestamp = pd.Timestamp.now()
df['dw_created_at'] = process_timestamp
df['dw_updated_at'] = process_timestamp

#Load df to db
engine = create_engine(ConnectionStrings().postgres_dw_stocks)
df.to_sql('dim_stocks', engine, schema = DbSchemas().dw_stocks, if_exists = 'append', index = False)
