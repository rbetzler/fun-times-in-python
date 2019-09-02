import pandas as pd
import scripts.ingestion.scraper as scraper
from scripts.sql_scripts.queries import scrape_yahoo_stocks


class YahooStockScraper(scraper.WebScraper):
        @property
        def get_urls(self) -> pd.DataFrame:
            urls = []
            tickers = ['AAPL', 'AA', 'KO', 'GE']

            for ticker in tickers:
                urls.append("https://query2.finance.yahoo.com/v8/finance/chart/"
                               + ticker + "?formatted=true&crumb=3eIGSD3T5Ul&lang=en-US&region=US&period1="
                               + '34450' + "&period2="
                               + '153145400400'
                               + "&interval=1d&events=div%7Csplit&corsDomain=finance.yahoo.com")

            return pd.DataFrame(urls)

        @property
        def query(self) -> str:
            return scrape_yahoo_stocks.QUERY

        @property
        def request_type(self) -> str:
            return 'json'

        @property
        def place_raw_file(self) -> bool:
            return True

        @property
        def export_folder(self) -> str:
            return '/mnt/audit/yahoo/stocks/'

        @property
        def export_file_name(self) -> str:
            return 'yahoo_test_'

        @property
        def table(self) -> str:
            return 'stocks'

        @property
        def schema(self) -> str:
            return 'yahoo'

        @property
        def n_cores(self) -> int:
            return 3

        @property
        def parallel_output(self) -> pd.DataFrame:
            output = pd.DataFrame({'open': [],
                                   'high': [],
                                   'low': [],
                                   'close': [],
                                   'volume': [],
                                   'dividend': [],
                                   'split_numerator': [],
                                   'split_denominator': [],
                                   'ticker': [],
                                   'dw_created_at': [],
                                   'dw_updated_at': []
                                   })
            return output

        @property
        def len_of_pause(self) -> int:
            return 5

        def parse(self, soup) -> pd.DataFrame:

            # check if data got returned
            assert type(soup.get('chart').get('error')) is not dict

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

            # join
            df = dates.merge(quotes, how='left', left_index=True, right_index=True)
            df = df.merge(adj_close, how='left', left_index=True, right_index=True)
            df = df.merge(dividends, how='left', on='date')
            df = df.merge(splits, how='left', on='date')

            # Convert unix timestamp to date time
            df['market_datetime'] = pd.to_datetime(df['date'], unit='s')
            df['ticker'] = ticker

            df['dw_created_at'] = self.run_date
            df['dw_updated_at'] = self.run_date

            return df


if __name__ == '__main__':
    YahooStockScraper().execute()

