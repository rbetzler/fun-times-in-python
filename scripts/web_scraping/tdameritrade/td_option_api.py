import datetime
import requests
import pandas as pd
from scripts.web_scraping import api_grabber
from scripts.utilities.db_utilities import ConnectionStrings, DbSchemas


class ApiGrabber(api_grabber.ApiGrabber):
    def __init__(self,
                 run_date=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'),
                 start_date=datetime.datetime.now().date().strftime('%Y-%m-%d'),
                 end_date=datetime.datetime.now().date().strftime('%Y-%m-%d')):
        self.db_connection = ConnectionStrings().postgres_dw_stocks
        self.run_date = run_date
        self.start_date = start_date
        self.end_date = end_date


    @property
    def get_api_calls(self) -> pd.DataFrame:
        api = self.api_call_base \
              + '?apikey=' + self.apikey \
              + '&symbol=' + self.tickers \
              + '&contractType=' + self.contract_types
        return pd.DataFrame([api])

    @property
    def contract_types(self) -> str:
        return 'ALL'

    @property
    def tickers(self) -> str:
        return 'AAPL'

    @property
    def api_call_base(self) -> str:
        return 'https://api.tdameritrade.com/v1/marketdata/chains'

    @property
    def apikey(self) -> str:
        return ''

    @property
    def place_raw_file(self) -> bool:
        return True

    @property
    def export_folder(self) -> str:
        return '/Users/rbetzler/Desktop/testing/'

    @property
    def export_file_name(self) -> str:
        return 'td_api_'

    @property
    def export_file_type(self) -> str:
        return '.csv'

    @property
    def load_to_db(self) -> bool:
        return False

    @property
    def table(self) -> str:
        return ''

    @property
    def append_to_table(self) -> str:
        return 'append'

    @property
    def index(self) -> bool:
        return False

    @property
    def parallel_output(self) -> pd.DataFrame:
        return pd.DataFrame()

    @property
    def n_cores(self) -> int:
        return 1

    @property
    def len_of_pause(self) -> int:
        return 0

    def call_api(self, call) -> requests.Response:
        api_response = requests.get(call)
        return api_response

    def parse(self, res) -> pd.DataFrame:
        res = res.json()

        symbol = res.get('symbol')
        volatility = res.get('volatility')
        n_contracts = res.get('numberOfContracts')
        interest_rate = res.get('interestRate')
        calls = res.get('callExpDateMap')
        puts = res.get('putExpDateMap')

        pass

    def parallelize(self, api) -> pd.DataFrame:
        api_response = self.call_api(api)
        df = self.parse(api_response)
        return df


if __name__ == '__main__':
    ApiGrabber().execute()
