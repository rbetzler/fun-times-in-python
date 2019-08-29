import time
import datetime
import pandas as pd
from scripts.ingestion import api_grabber
from scripts.sql_scripts.queries import td_option_tickers


class TdOptionsApi(api_grabber.ApiGrabber):
    @property
    def get_api_calls(self) -> pd.DataFrame:
        apis = []
        tickers = []
        for idx, row in self.tickers.iterrows():
            apis.append(self.api_call_base
                        + row.values[0] + '/pricehistory'
                        + '?apikey=' + self.apikey
                        + '&periodType=' + self.period_type
                        + '&period=' + self.period
                        + '&frequencyType=' + self.frequency_type
                        + '&frequency=' + self.frequency)
            tickers.append(row.values[0])
        df = pd.DataFrame(data=apis, index=tickers)
        return df

    @property
    def contract_types(self) -> str:
        return 'ALL'

    @property
    def query(self) -> str:
        return td_option_tickers.QUERY.format(
            batch_size=self.batch_size,
            batch_start=self.lower_bound
        )

    @property
    def tickers(self) -> pd.DataFrame:
        df = self.get_call_inputs_from_db
        return df

    @property
    def api_call_base(self) -> str:
        return 'https://api.tdameritrade.com/v1/marketdata/'

    @property
    def apikey(self) -> str:
        return 'B41S3HBMUXQOLM81JXQ7CWXJMSN17CSM'

    @property
    def period_type(self) -> str:
        return 'year'

    @property
    def period(self) -> str:
        return '2'

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
        return 1

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
        df['market_datetime'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(df['datetime'].values[0] / 1000))

        df = df.rename(columns=self.column_renames())
        return df


if __name__ == '__main__':
    batch_size = 100
    n_batches = 30
    for batch in range(1, n_batches):
        lower_bound = (batch-1) * batch_size
        print('Beginning Batch: ' + str(batch))
        TdOptionsApi(lower_bound=lower_bound, batch_size=batch_size).execute()
        print('Completed Batch: ' + str(batch))
