#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 23:02:27 2019

@author: nautilus
"""

#Import py functions
import sys
import psycopg2
import datetime
import pandas as pd
import multiprocessing as mp
from sqlalchemy import create_engine

#Import custom function
sys.path.append('/home/py-scripts/web-scraping/scrapers')
from scrape_yahoo import ExtractYahooStockPerformance

sys.path.append('/home/py-scripts/utilities')
from db_utilities import ConnectionStrings, DbSchemas

print('Start time: ' + str(datetime.datetime.now()))

#Grab tickers
conn_string = ConnectionStrings().postgres_dw_stocks
conn = psycopg2.connect(conn_string)
cursor = conn.cursor()

sql_file = '/home/sql-scripts/queries/scrape_stocks.sql'
query = open(sql_file).read()

cursor.execute(query)
tickers = []
for row in cursor:
    tickers.append(row)

tickers = tickers[:10]

outputs = pd.DataFrame({'open' : [],
                       'high' : [],
                       'low' : [],
                       'close' : [],
                       'adj_close' : [],
                       'volume' : [],
                       'unix_timestamp' : [],
                       'date_time' : [],
                       'dividend' : [],
                       'split_numerator' : [],
                       'split_denominator' : [],
                       'ticker' : []
                       })

outputs = mp.Pool(4).map(ExtractYahooStockPerformance, tickers)
output = pd.concat(outputs)

#Load df to db
engine = create_engine(ConnectionStrings().postgres)
output.to_sql('fact_yahoo_stocks', engine, schema = DbSchemas().dw_stocks, if_exists = 'append')
