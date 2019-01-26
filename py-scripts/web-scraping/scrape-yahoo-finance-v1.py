#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 19:45:20 2018

@author: rbetzler
"""

import re
import time
import requests
import psycopg2
import datetime
import numpy as np
import pandas as pd
import multiprocessing as mp
from sqlalchemy import create_engine

#Loop x parallel execution

conn_string = "host='localhost' dbname='postgres' user='postgres' password='D@t@_Mgmt'"
conn = psycopg2.connect(conn_string)
cursor = conn.cursor()

query = (" select count(distinct(ticker)) from dim_directory_stocks "
         " where ticker !~ '[\^.~]' and character_length(ticker) between 1 and 4 ;")

cursor.execute(query)

cntTickers = cursor.fetchone()[0]

#nGroups = 500

#segments = round(cntTickers/nGroups)

#lowerBound = 0
#upperBound = segments

#for n in range(0, nGroups):
#for n in range(0, 100):

    #Pull tickers

conn_string = "host='localhost' dbname='postgres' user='postgres' password='D@t@_Mgmt'"
conn = psycopg2.connect(conn_string)
cursor = conn.cursor()

query = (" select ticker from "
         " (select ticker, row_number() over (order by ticker) as cnt "
         " from dim_directory_stocks where ticker !~ '[\^.~]' "
         " and character_length(ticker) between 1 and 4 "
         " and ticker not in (select distinct ticker from dim_yahoo_stocks)) as sub ")
#             " where cnt between " + str(lowerBound) + " and " + str(upperBound) + " ")

cursor.execute(query)

tickers = []
for row in cursor:
    tickers.append(row[0])

missingTickers = []

if len(tickers) > 0:

    #Shell df
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
    pwd = ''
    engine = create_engine('postgresql://postgres:' + pwd + '@localhost:5432/postgres')
    output.to_sql('dim_yahoo_stocks', engine, if_exists = 'append')

#        print("Appending data to db. Time: " + str(datetime.datetime.now()))


#Set bounds for next loop
#lowerBound = lowerBound + segments + 1
#upperBound = upperBound + segments + 1









#==============================================================================
#
# ALTERNATIVE METHOD
#
# #Date parameters
# datePast = "34450"
# datePresent = "153145400400"
#
# #Loop extract over tickers
# for ticker in tickers:
#     output = output.append(ExtractYahooStockPerformance(ticker, datePast, datePresent))
#
# #Parallel process -- not ready yet -- need to complete sql write in function
# with Pool(8) as p: p.map(ExtractYahooStockPerformance, tickers)
#
# #Append http data to df
# outputs = pd.DataFrame(output)
#
#==============================================================================
