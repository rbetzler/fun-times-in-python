import pandas as pd
from scripts.ingestion import api_grabber
from scripts.sql_scripts.queries import td_option_tickers


class TDOptionsAPI(api_grabber.APIGrabber):
    def format_api_calls(self, idx, row) -> tuple:
        api_call = 'https://api.tdameritrade.com/v1/marketdata/' \
                   + row.values[0] + '/pricehistory' \
                   + '?apikey=' + self.api_secret \
                   + '&periodType=' + self.period_type \
                   + '&period=' + self.period \
                   + '&frequencyType=' + self.frequency_type \
                   + '&frequency=' + self.frequency
        api_name = row.values[0]
        return api_call, api_name

    @property
    def contract_types(self) -> str:
        return 'ALL'

    @property
    def api_calls_query(self) -> str:
        return td_option_tickers.QUERY.format(
            batch_size=self.batch_size,
            batch_start=self.lower_bound
        )

    @property
    def api_name(self) -> str:
        return 'API_TD'

    @property
    def period_type(self) -> str:
        return 'year'

    @property
    def period(self) -> str:
        return '20'

    @property
    def frequency_type(self) -> str:
        return 'daily'

    @property
    def frequency(self) -> str:
        return '1'

    @property
    def export_folder(self) -> str:
        folder = 'audit/processed/td_ameritrade/equities/' \
                 + self.run_time.strftime('%Y_%m_%d_%H_%S') \
                 + '/'
        return folder

    @property
    def export_file_name(self) -> str:
        return 'td_equities_'

    @property
    def place_raw_file(self) -> bool:
        return True

    @property
    def load_to_db(self) -> bool:
        return False

    @property
    def table(self) -> str:
        return ''

    @property
    def n_workers(self) -> int:
        return 15

    @property
    def len_of_pause(self) -> int:
        return 5

    def column_renames(self) -> dict:
        names = {
            'datetime': 'market_datetime_epoch'
        }
        return names

    def parse(self, res) -> pd.DataFrame:
        res = res.json()

        symbol = res.get('symbol')
        empty = res.get('empty')
        candles = pd.DataFrame(res.get('candles'))

        df = candles
        df['symbol'] = symbol
        df['empty'] = empty
        try:
            df['market_datetime'] = pd.to_datetime(df['datetime'], unit='ms')
        except KeyError:
            print('ehh')

        df = df.rename(columns=self.column_renames())
        return df


if __name__ == '__main__':
    batch_size = 100
    n_batches = 30
    for batch in range(1, n_batches):
        lower_bound = (batch-1) * batch_size
        print('Beginning Batch: ' + str(batch))
        TDOptionsAPI(lower_bound=lower_bound, batch_size=batch_size).execute()
        print('Completed Batch: ' + str(batch))
