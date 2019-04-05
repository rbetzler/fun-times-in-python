#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 22:42:20 2019

@author: nautilus
"""

class ConnectionStrings:

    def __init__(self):

        self.postgres = 'postgresql://postgres:password@localhost:5432/dw_stocks'

        self.postgres_default = 'postgresql://postgres:password@localhost:5432'

class DbSchemas:

    def __init__(self):

        self.dw = 'dw'

        self.dw_stocks = 'dw_stocks'
