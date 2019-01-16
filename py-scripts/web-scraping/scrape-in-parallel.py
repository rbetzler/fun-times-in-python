#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 22:17:41 2019

@author: nautilus
"""

import os
import time
import psycopg2
import pandas as pd
from urllib.request import urlretrieve
from multiprocessing.dummy import Pool


def loop_urls(https_request):
    try:
        print("Downloading %s to %s..." % (https_request, file_name) )
        urlretrieve(https_request, file_path)
        time.sleep(4)
        print("Done.")
    except:
        return
    
with Pool(8) as p: p.map(loop_urls, urls)