import re
import numpy as np
import pandas as pd
import scripts.web_scraping.scraper as scraper


class YahooStockScraper(scraper.WebScraper):
        @property
        def get_urls(self) -> pd.DataFrame:
            # urls = self.get_urls_from_db
            #
            # urls['url'] = "https://query2.finance.yahoo.com/v8/finance/chart/" \
            #               + urls['ticker'] + "?formatted=true&crumb=3eIGSD3T5Ul&lang=en-US&region=US&period1=" \
            #               + urls['start_date'] + "&period2=" \
            #               + urls['end_date'] \
            #               + "&interval=1d&events=div%7Csplit&corsDomain=finance.yahoo.com"
            #
            # return urls['url']

            return pd.DataFrame(["https://query2.finance.yahoo.com/v8/finance/chart/" \
                                 + 'AAPL' + "?formatted=true&crumb=3eIGSD3T5Ul&lang=en-US&region=US&period1=" \
                                 + '34450' + "&period2=" \
                                 + '153145400400' \
                                 + "&interval=1d&events=div%7Csplit&corsDomain=finance.yahoo.com"])

        @property
        def sql_file(self) -> str:
            return '/home/sql_scripts/queries/scrape_stocks.sql'

        @property
        def request_type(self) -> str:
            return 'json'

        @property
        def drop_raw_file(self) -> bool:
            return True

        @property
        def file_path(self) -> str:
            return '/Users/rickbetzler/Desktop/yahoo_test.csv'

        @property
        def table(self) -> str:
            return 'fact_yahoo_stocks'

        @property
        def n_cores(self) -> int:
            return 1

        @property
        def parallel_output(self) -> pd.DataFrame:
            output = pd.DataFrame({'open' : [],
                                   'high' : [],
                                   'low' : [],
                                   'close' : [],
                                   'adj_close' : [],
                                   'volume' : [],
                                   'unix_timestamp' : [],
                                   'date_time' : [],
                                   'dividend' : [],
                                   'split_numerator' : [],
                                   'split_denominator' : [],
                                   'ticker' : [],
                                   'dw_created_at' : [],
                                   'dw_updated_at' : []
                                   })
            return output

        @property
        def len_of_pause(self) -> int:
            return 6

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

            return df


if __name__ == '__main__':
    YahooStockScraper().execute()

