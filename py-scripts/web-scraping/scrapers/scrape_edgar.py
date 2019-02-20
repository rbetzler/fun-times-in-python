#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 16:56:11 2019

@author: nautilus
"""

import requests
import pandas as pd

def ConvertFileHtmPathToTxt(df, company_id, file_path):
    
    df['url'] = ('http://www.sec.gov/Archives/edgar/data/' 
      + df[company_id] 
      + '/'
      + df[file_path].str.split('/', expand = True)[7].str.rsplit('-', expand = True, n = 1)[0].str.replace('-', '') 
      + '/'
      + df[file_path].str.split('/', expand = True)[7].str.rsplit('-', expand = True, n = 1)[0]
      + '.txt')
    
    return df

def ExtractEdgarIndex(year, quarter, date):
    
    #Get crawler index
    url = ('https://www.sec.gov/Archives/edgar/daily-index/' 
           + year + '/' + quarter + '/crawler.' + date + '.idx')
    edgar_index = requests.get(url).text
    
    #Covert to list
    edgar_index = edgar_index.split('\n')
    
    #Drop header
    for row in range(0, len(edgar_index)):
        characters = set(edgar_index[row])
        if characters == {'-'}:
            edgar_index = edgar_index[row+1:]
            break
    
    #Convert to pandas
    df_edgar = pd.Series(data = edgar_index)
    
    #Partition positions
    part_one = 62
    part_two = 74
    part_three = 86
    part_four = 98
    
    #Split to columns
    df = pd.DataFrame({
            'company' : df_edgar.str.slice(stop = part_one).str.strip(),
            'filing_type' : df_edgar.str.slice(start = part_one, stop = part_two).str.strip(),
            'company_id' : df_edgar.str.slice(start = part_two, stop = part_three).str.strip(),
            'date' : df_edgar.str.slice(start = part_three, stop = part_four).str.strip(),
            'file_path' : df_edgar.str.slice(start = part_four).str.strip()})
    
    df = ConvertFileHtmPathToTxt(df, 'company_id', 'file_path')
    
    return df


##Example
#year = '2019'
#quarter = 'QTR1'
#date = '20190205'
#example = ExtractEdgarIndex(year, quarter, date)
