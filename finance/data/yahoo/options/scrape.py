import pandas as pd
from finance.data import scraper


class Options(scraper.Caller):
    @property
    def job_name(self) -> str:
        return 'yahoo_options'

    @property
    def request_type(self) -> str:
        return 'json'

    @property
    def requests_query(self) -> str:
        query = r'''
            select distinct ticker
            from nasdaq.listed_stocks
            where ticker !~ '[\^.~]'
                and character_length(ticker) between 1 and 4
            limit 15;
            '''
        return query

    def format_requests(self, row) -> tuple:
        key = row.ticker
        request = f'https://query2.finance.yahoo.com/v8/finance/chart/{key}?formatted=true&crumb=3eIGSD3T5Ul&lang=en-US&region=US&period1=34450&period2=153145400400&interval=1d&events=div%7Csplit&corsDomain=finance.yahoo.com'
        return key, request

    @property
    def export_folder(self) -> str:
        return f'audit/yahoo/options/{self.folder_datetime}/'

    @property
    def export_file_name(self) -> str:
        return 'yahoo_options_'

    @property
    def len_of_pause(self) -> int:
        return 5

    def parse(self, soup, call) -> pd.DataFrame:

        if type(soup.get('chart').get('error')) is not dict:

            ticker = soup.get('chart').get('result')[0].get('meta').get('symbol')

            # dates
            dates = pd.DataFrame(soup.get('chart').get('result')[0].get('timestamp'))
            dates.columns = ['date']

            # open close low high vol
            quotes = soup.get('chart').get('result')[0].get('indicators').get('quote')[0]
            quotes = pd.DataFrame.from_dict(quotes)

            # adjusted close
            adj_close = soup.get('chart').get('result')[0].get('indicators').get('adjclose')[0]
            adj_close = pd.DataFrame.from_dict(adj_close)

            # dividends and splits
            dividends = soup.get('chart').get('result')[0].get('events').get('dividends')
            dividends = pd.DataFrame((dividends.values()))
            dividends = dividends.rename(columns={'amount': 'dividend'})

            # splits
            splits = soup.get('chart').get('result')[0].get('events').get('splits')
            splits = pd.DataFrame((splits.values()))
            splits = splits.rename(columns={'numerator': 'split_numerator',
                                            'denominator': 'split_denominator',
                                            'splitRatio': 'split_ratio'})

            df = dates.merge(quotes, how='left', left_index=True, right_index=True)
            df = df.merge(adj_close, how='left', left_index=True, right_index=True)
            df = df.merge(dividends, how='left', on='date')
            df = df.merge(splits, how='left', on='date')

            df['market_datetime'] = pd.to_datetime(df['date'], unit='s')
            df['ticker'] = ticker

        else:
            df = pd.DataFrame()

        return df


if __name__ == '__main__':
    Options().execute()

