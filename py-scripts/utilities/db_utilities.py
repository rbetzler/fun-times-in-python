#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 22:42:20 2019

@author: nautilus
"""

class ConnectionStrings:

    def __init__(self):

        self.postgres_default = 'postgresql://postgres:password@172.18.0.3:5432/postgres'

        self.postgres_dw_stocks = 'postgresql://postgres:password@172.18.0.3:5432/dw_stocks'

class DbSchemas:

    def __init__(self):

        self.dw = 'dw'

        self.dw_stocks = 'dw_stocks'
