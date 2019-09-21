import re
import time
import pandas as pd
from finance.ingestion import scraper


class CashFlowsScraper(scraper.Caller):
    # general
    @property
    def job_name(self) -> str:
        return 'yahoo_cash_flows'

    # calls
    @property
    def calls_query(self) -> str:
        query = f"select distinct ticker from nasdaq.listed_stocks "\
                f"where ticker !~ '[\^.~]' and character_length(ticker) between 1 and 4 limit 10;"
        return query

    def format_calls(self, idx, row) -> tuple:
        company = row['ticker']
        url_prefix = 'https://finance.yahoo.com/quote/'
        url_suffix = '/cash-flow?p='
        url = url_prefix + company + url_suffix + company
        return url, company

    # db
    @property
    def load_to_db(self) -> bool:
        return True

    @property
    def table(self) -> str:
        return 'cash_flows'

    @property
    def schema(self) -> str:
        return 'yahoo'

    # parse
    def parse(self, soup, company) -> pd.DataFrame:
        dates = []
        tuples = []
        cnt_vals = 0
        table = soup.find('table')
        for span in table.find_all('span'):
            text = span.text
            if re.match("^\d{2}\/\d{2}\/\d{4}$", text):
                dates.append(text)
            elif re.match("^[A-Za-z.^,/'\s_-]+$", text):
                name = text
                cnt_vals = 0
            elif re.match("^(\d|-)?(\d|,)*\.?\d*$", text):
                tuples.append((name, dates[cnt_vals], text))
                cnt_vals = cnt_vals + 1
        df = pd.DataFrame(tuples, columns=['metric', 'date', 'val'])
        df['val'] = df['val'].str.replace(',', '').astype(float)
        df['ticker'] = company
        return df

    def parallelize(self, url) -> pd.DataFrame:
        soup = self.summon(url[0])
        df = self.parse(soup, url.name)
        time.sleep(self.len_of_pause)
        return df


if __name__ == '__main__':
    CashFlowsScraper().execute()
