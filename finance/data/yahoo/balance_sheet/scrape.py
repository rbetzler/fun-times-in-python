import re
import pandas as pd
from finance.data import scraper


# Left off here, refactoring
class BalanceSheet(scraper.Caller):
    @property
    def job_name(self) -> str:
        return 'yahoo_balance_sheet'

    @property
    def export_folder(self) -> str:
        folder = f'audit/yahoo/balance_sheet/{self.folder_datetime}/'
        return folder

    @property
    def export_file_name(self) -> str:
        return 'yahoo_balance_sheet_'

    @property
    def requests_query(self) -> str:
        query = '''
            select distinct ticker
            from nasdaq.listed_stocks
            where ticker !~ '[\^.~]'
                and character_length(ticker) between 1 and 4
            limit 10;
            '''
        return query

    def format_requests(self, row) -> tuple:
        key = row.ticker
        request = f'https://finance.yahoo.com/quote/{key}/balance-sheet?p={key}'
        return key, request

    def parse(self, response, call) -> pd.DataFrame:
        dates = []
        tuples = []
        cnt_vals = 0
        table = response.find('table')
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
        df['ticker'] = call
        return df


if __name__ == '__main__':
    BalanceSheet().execute()
