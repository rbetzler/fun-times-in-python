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

print('Start time: ' + str(datetime.datetime.now()))

#def ExtractYahooStockPerformance(tkr, dateStart, dateEnd):
def ExtractYahooStockPerformance(tkr):

    dateStart = "34450"
    dateEnd = "153145400400"

    url = ("https://query2.finance.yahoo.com/v8/finance/chart/"
           + tkr + "?formatted=true&crumb=3eIGSD3T5Ul&lang=en-US&region=US&"
           "period1=" + dateStart + "&period2=" + dateEnd +
           "&interval=1d&events=div%7Csplit&corsDomain=finance.yahoo.com")

    webRaw = requests.get(url).json()

    web = str(webRaw)

    if web == "{'chart': {'result': None, 'error': {'code': 'Not Found', 'description': 'No data found, symbol may be delisted'}}}":

        print('Ticker not found. ')

        missingTickers.append(tkr)

        df = pd.DataFrame({'open' : [],
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

    else:

        partitions = re.split("\[|\]|\{|\}", web)


        #Create appendable strings and ids for scanning divs and splits
        dividends = ''
        splits = ''
        activeDividends = 0
        activeSplits = 0


        for n in range(0, len(partitions)):
            if partitions[n].find("open") != -1:
                opn = partitions[n + 1]
        #        print("open")

            elif partitions[n].find("high") != -1:
                high = partitions[n + 1]
        #        print("high")

            elif partitions[n].find("low") != -1:
                low = partitions[n + 1]
        #        print("low")

            elif partitions[n].find("'close") != -1:
                close = partitions[n + 1]
        #        print("close")

            elif partitions[n].find("adjclose") != -1:
                closeAdj = partitions[n + 1]
        #        print("adj close")

            elif partitions[n].find("volume") != -1:
                volume = partitions[n + 1]
        #        print("vol")
            elif partitions[n].find("timestamp") != -1:
                timestamp = partitions[n + 1]
        #        print("timestamp")

            #Append divs or splits
            if activeDividends == 1 and partitions[n].find('date') != -1:
                dividends = dividends + partitions[n]
            if activeSplits == 1 and partitions[n].find('date') != -1:
                splits = splits + partitions[n]

            #Flip if scanning divs or splits
            if partitions[n].find("dividends") != -1:
                activeDividends = 1
            if partitions[n].find("splits") != -1:
                activeSplits = 1

            #Reset div and split ids
            if partitions[n] == '':
                activeDividends = 0
                activeSplits = 0
        #        print(partitions[n-1][:100])


        #Split strings into lists
        opn = opn.split(",")
        high = high.split(",")
        low = low.split(",")
        close = close.split(",")
        closeAdj = closeAdj.split(",")
        volume = volume.split(",")
        timestamp = timestamp.split(",")

        #Convert lists to pandas
        df = pd.DataFrame({'open' : opn,
                           'high' : high,
                           'low' : low,
                           'close' : close,
                           'adj_close' : closeAdj,
                           'volume' : volume,
                           'unix_timestamp' : timestamp})

        #Strip all strings
        df = df.apply(lambda x: x.str.strip())

        #Conver unix timestamp to date time
        df['date_time'] = pd.to_datetime(df['unix_timestamp'], unit = 's')

        #Replace None string with Nan
        df = df.replace("None", np.NaN)

        #Convert strings to floats
        dfObj = df.select_dtypes(['object'])
        df[dfObj.columns] = dfObj.apply(lambda x: x.astype(float))


        #Process dividends

        dividends = dividends.split("'amount':")

        divAmount = []
        divDate = []

        for n in range(1, len(dividends)):
            cuts = dividends[n].split("'date':")
            divAmount.append(cuts[0])
            divDate.append(cuts[1])

        dfDividends = pd.DataFrame({'date_dividend' : pd.Series(divDate, dtype = str),
                                    'dividend' : pd.Series(divAmount, dtype = str)
                                    })

        #Convert unix timestamp to date time
        dfDividends['date_dividend'] = pd.to_datetime(dfDividends['date_dividend'], unit = 's')

        #Convert div string to float
        dfDividends.dividend = dfDividends.dividend.str.replace(",", "")
        dfDividends.dividend = dfDividends.dividend.str.strip()
        dfDividends.dividend  = dfDividends.dividend.astype(float)


        #Process splits

        splits = splits.split("'date':")

        dates = []
        numerators = []
        denominators = []

        for n in range(1, len(splits)):
            cuts = splits[n].split(",")

            date = cuts[0]
            numerator = cuts[1].split(":")[1]
            denominator = cuts[2].split(":")[1]

            dates.append(date)
            numerators.append(numerator)
            denominators.append(denominator)

        dfSplits = pd.DataFrame({'date_split' : dates,
                                 'split_numerator' : pd.Series(numerators, dtype = int),
                                 'split_denominator' : pd.Series(denominators, dtype = int)
                                 })

        dfSplits['date_split'] = pd.to_datetime(dfSplits['date_split'], unit = 's')


        #Join df with dividends and splits

        df = df.join(dfDividends.set_index('date_dividend'),
                     how = 'outer', on = 'date_time', rsuffix = '_divs')

        df = df.join(dfSplits.set_index('date_split'),
                     how = 'outer', on = 'date_time', rsuffix = '_splits')

        df['ticker'] = tkr

#        print("Processing complete. Sleeping.")

        time.sleep(6)

#        print("Finished scraping ticker: " + str(tkr))

    return df




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
    engine = create_engine('postgresql://postgres:D@t@_Mgmt@localhost:5432/postgres')
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
