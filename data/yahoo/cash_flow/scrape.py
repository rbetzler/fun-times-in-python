import re
import pandas as pd
from data import scraper


class CashFlow(scraper.Caller):
    @property
    def job_name(self) -> str:
        return 'yahoo_cash_flow'

    @property
    def export_folder(self) -> str:
        folder = f'audit/yahoo/cash_flow/{self.folder_datetime}/'
        return folder

    @property
    def export_file_name(self) -> str:
        return 'yahoo_cash_flow_'

    @property
    def requests_query(self) -> str:
        query = r'''
            select distinct ticker
            from nasdaq.listed_stocks
            where ticker !~ '[\^.~]'
                and character_length(ticker) between 1 and 4
            limit 10;
            '''
        return query

    def format_requests(self, row) -> tuple:
        key = row.ticker
        request = f'https://finance.yahoo.com/quote/{key}/cash-flow?p={key}'
        return key, request

    def parse(self, soup, company) -> pd.DataFrame:
        dates = []
        tuples = []
        cnt_vals = 0
        table = soup.find('table')

        if table:
            for span in table.find_all('span'):
                text = span.text
                if re.match(r'^\d{2}\/\d{2}\/\d{4}$', text):
                    dates.append(text)
                elif re.match(r"^[A-Za-z.^,/'\s_-]+$", text):
                    name = text
                    cnt_vals = 0
                elif re.match(r'^(\d|-)?(\d|,)*\.?\d*$', text):
                    tuples.append((name, dates[cnt_vals], text))
                    cnt_vals = cnt_vals + 1
            df = pd.DataFrame(tuples, columns=['metric', 'date', 'val'])
            df['val'] = df['val'].str.replace(',', '').astype(float)
            df['ticker'] = company

        else:
            df = pd.DataFrame()
        return df


if __name__ == '__main__':
    CashFlow().execute()
