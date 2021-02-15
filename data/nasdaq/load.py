""" Listed stocks on all exchanges from nasdaq website
Download csv from: https://www.nasdaq.com/market-activity/stocks/screener
"""

import datetime
import pandas as pd
from data import loader


class FileIngestion(loader.FileIngestion):
    @property
    def job_name(self) -> str:
        return 'nasdaq_listed_stocks'

    @property
    def directory(self) -> str:
        return 'nasdaq/listed_stocks'

    @property
    def import_file_prefix(self) -> str:
        return 'nasdaq_screener'

    @property
    def table(self) -> str:
        return 'listed_stocks'

    @property
    def schema(self) -> str:
        return 'nasdaq'

    @property
    def column_mapping(self) -> dict:
        cols = {
            'Symbol': 'ticker',
            'ADR TSO': 'adr_tso',
            'Name': 'company',
            'IPO Year': 'ipo_year',
            'Sector': 'sector',
            'Industry': 'industry'
        }
        return cols


if __name__ == '__main__':
    FileIngestion().execute()
