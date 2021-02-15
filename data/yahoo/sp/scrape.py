import io
import pandas as pd
import requests
from data import scraper
from typing import List, Tuple


class YahooSPCaller(scraper.Caller):

    @property
    def job_name(self) -> str:
        return 'sp'

    @property
    def calls_query(self):
        pass

    def format_calls(self, row):
        pass

    def get_calls(self) -> List[Tuple]:
        key = ''
        req = 'https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC?period1=-1325635200&period2=1581206400&interval=1d&events=history&includeAdjustedClose=true'
        return [(key, req)]

    @property
    def export_folder(self) -> str:
        return f'audit/yahoo/{self.job_name}/{self.folder_datetime}/'

    @property
    def export_file_name(self) -> str:
        return f'{self.job_name}'

    def parse(
            self,
            response: requests.Response,
            key: str,
    ) -> pd.DataFrame:
        df = pd.read_csv(io.StringIO(response.text))
        return df


if __name__ == '__main__':
    YahooSPCaller().execute()
