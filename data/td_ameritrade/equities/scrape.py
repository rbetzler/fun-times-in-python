import pandas as pd
from data import scraper


class TDOptionsAPI(scraper.Caller):
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
        request = f'https://api.tdameritrade.com/v1/marketdata/{key}/pricehistory?apikey={self.api_secret}&periodType={self.period_type}&period={self.period}&frequencyType={self.frequency_type}&frequency={self.frequency}'
        return key, request

    @property
    def period_type(self) -> str:
        return 'year'

    @property
    def period(self) -> str:
        return '1'

    @property
    def frequency_type(self) -> str:
        return 'daily'

    @property
    def frequency(self) -> str:
        return '1'

    @property
    def export_folder(self) -> str:
        return f'audit/td_ameritrade/equities/{self.folder_datetime}/'

    @property
    def export_file_name(self) -> str:
        return 'td_equities_'

    @property
    def n_workers(self) -> int:
        return 15

    @property
    def len_of_pause(self) -> int:
        return 10

    @property
    def column_renames(self) -> dict:
        names = {'datetime': 'market_datetime_epoch'}
        return names

    def parse(self, res, key) -> pd.DataFrame:
        res = res.json()

        symbol = res.get('symbol')
        empty = res.get('empty')
        candles = pd.DataFrame(res.get('candles'))

        df = candles
        df['symbol'] = symbol
        df['empty'] = empty

        df['market_datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        df = df.rename(columns=self.column_renames)
        return df


if __name__ == '__main__':
    batch_size = 5000
    n_batches = 2
    for batch in range(1, n_batches):
        lower_bound = (batch - 1) * batch_size
        print('Beginning Batch: ' + str(batch))
        TDOptionsAPI(lower_bound=lower_bound, batch_size=batch_size).execute()
        print('Completed Batch: ' + str(batch))
