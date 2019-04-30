#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:45:16 2019

@author: nautilus
"""

#Import py functions
import os
import sys
import psycopg2
import datetime
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

#Import custom function
sys.path.append('/home/utilities')
from db_utilities import ConnectionStrings

print('Start time: ' + str(datetime.datetime.now()))

databases = []
directory = '/home/sql-scripts/databases'
for file in os.listdir(directory):
     databases.append(open(directory + '/' + file).read())

schemas = []
directory = '/home/sql-scripts/schemas'
for file in os.listdir(directory):
     schemas.append(open(directory + '/' + file).read())

tables = []
directory = '/home/sql-scripts/ddl'
for file in os.listdir(directory):
     tables.append(open(directory + '/' + file).read())


#Connect to default db to create dw_stocks
conn_string = ConnectionStrings().postgres_default
conn = psycopg2.connect(conn_string)
cursor = conn.cursor()

#Isolate commit
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

for database in databases:
    cursor.execute(database)
    conn.commit()

conn.close()
cursor.close()


#Connect to dw_stocks to create schemas
conn_string = ConnectionStrings().postgres_dw_stocks
conn = psycopg2.connect(conn_string)
cursor = conn.cursor()

#Create schemas
for schema in schemas:
    cursor.execute(schema)
    conn.commit()

#Create tables
for table in tables:
    cursor.execute(table)
    conn.commit()

conn.close()
cursor.close()

print('End time: ' + str(datetime.datetime.now()))
