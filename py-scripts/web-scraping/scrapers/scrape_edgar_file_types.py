#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 22:02:35 2019

@author: nautilus
"""

import bs4
import requests
import pandas as pd

#Get sec form web page
raw_html = requests.get('https://www.sec.gov/forms').text

#Convert to soup
soup = bs4.BeautifulSoup(raw_html)

#Use tags to get file types and descriptions
soup_file_types = soup.find_all('td', {'class' : 'release-number-content views-field views-field-field-release-number is-active'})
soup_descriptions = soup.find_all('td', {'class' : 'display-title-content views-field views-field-field-display-title'})

#Load into lists
file_types = []
for row in soup_file_types:
    file_types.append(row.get_text().replace('Number:', '').strip())

file_descriptions = []    
for row in soup_descriptions:
    file_descriptions.append(row.get_text().replace('Description:', '').strip())

#Convert lists to pandas
df_files = pd.DataFrame({'filing_type' : file_types, 'description' : file_descriptions})
