#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 21:45:16 2019

@author: nautilus
"""

#Import py functions
import os
import psycopg2
import datetime

print('Start time: ' + str(datetime.datetime.now()))

schemas = []
directory = '/home/nautilus/development/fun-times-in-python/sql-scripts/schemas'
for file in os.listdir(directory):
     schemas.append(open(directory + '/' + file).read())

tables = []
directory = '/home/nautilus/development/fun-times-in-python/sql-scripts/ddl'
for file in os.listdir(directory):
     tables.append(open(directory + '/' + file).read())

#Connect
conn_string = "host='10.152.183.137' dbname='dw_stocks' user='dbadmin' password='password' port='5432'"
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
