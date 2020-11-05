import pandas as pd
from finance.data import scraper


class TDQuotesAPI(scraper.Caller):
    @property
    def job_name(self) -> str:
        return 'API_TD'

    @property
    def calls_query(self) -> str:
        query = r'''
            SELECT DISTINCT ticker
            FROM nasdaq.listed_stocks
            WHERE ticker !~ '[\^.~]'
                AND CHARACTER_LENGTH(ticker) BETWEEN 1 AND 4
            LIMIT {batch_size}
            OFFSET {batch_start}
            '''
        return query.format(batch_size=self.batch_size, batch_start=self.lower_bound)

    def format_calls(self, row) -> tuple:
        key = row.ticker
        request = f'https://api.tdameritrade.com/v1/marketdata/{key}/quotes?apikey={self.api_secret}'
        return key, request

    @property
    def export_folder(self) -> str:
        return f'audit/td_ameritrade/quotes/{self.folder_datetime}/'

    @property
    def export_file_name(self) -> str:
        return 'td_quotes_'

    @property
    def n_workers(self) -> int:
        return 15

    @property
    def len_of_pause(self) -> int:
        return 10

    @property
    def column_renames(self) -> dict:
        names = {
            'assetType': 'asset_type',
            'assetMainType': 'asset_main_type',
            'bidPrice': 'bid_price',
            'bidSize': 'bid_size',
            'bidId': 'bid_id',
            'askPrice': 'ask_price',
            'askSize': 'ask_size',
            'askId': 'ask_id',
            'lastPrice': 'last_price',
            'lastSize': 'last_size',
            'lastId': 'last_id',
            'openPrice': 'open_price',
            'highPrice': 'high_price',
            'lowPrice': 'low_price',
            'bidTick': 'bid_tick',
            'closePrice': 'close_price',
            'netChange': 'net_change',
            'totalVolume': 'total_volume',
            'quoteTimeInLong': 'quote_time_in_long',
            'tradeTimeInLong': 'trade_time_in_long',
            'exchangeName': 'exchange_name',
            '52WkHigh': '52_week_high',
            '52WkLow': '52_week_low',
            'nAV': 'nav',
            'peRatio': 'pe_ratio',
            'divAmount': 'dividend_amount',
            'divYield': 'dividend_yield',
            'divDate': 'dividend_date',
            'securityStatus': 'security_status',
            'regularMarketLastPrice': 'regular_market_last_price',
            'regularMarketLastSize': 'regular_market_last_size',
            'regularMarketNetChange': 'regular_market_net_change',
            'regularMarketTradeTimeInLong': 'regular_market_trade_time_in_long',
            'netPercentChangeInDouble': 'net_percent_change_in_double',
            'markChangeInDouble': 'mark_change_in_double',
            'markPercentChangeInDouble': 'mark_percent_change_in_double',
            'regularMarketPercentChangeInDouble': 'regular_market_percent_change_in_double',
            }
        return names

    def parse(self, res, key) -> pd.DataFrame:
        res = res.json()
        try:
            df = pd.DataFrame(res.get(key), index=[0])
            df = df.rename(columns=self.column_renames)
            for col in ['quote_time_in_long', 'trade_time_in_long', 'regular_market_trade_time_in_long']:
                df[col + '_datetime'] = pd.to_datetime(df[col], unit='ms')
        except Exception as e:
            print(f'Failed on {key} with error {e}')
            df = pd.DataFrame(columns=self.column_renames.values())
        return df


if __name__ == '__main__':
    batch_size = 5000
    n_batches = 2
    for batch in range(1, n_batches):
        lower_bound = (batch - 1) * batch_size
        print('Beginning Batch: ' + str(batch))
        TDQuotesAPI(lower_bound=lower_bound, batch_size=batch_size).execute()
        print('Completed Batch: ' + str(batch))
