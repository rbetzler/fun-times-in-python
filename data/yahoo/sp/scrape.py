import datetime
import io
import pandas as pd
import requests
from utilities import utils


class YahooSPCaller:
    def __init__(self):
        self.run_datetime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    @property
    def call(self) -> str:
        return 'https://query1.finance.yahoo.com/v7/finance/download/%5EGSPC?period1=-1325635200&period2=1581206400&interval=1d&events=history&includeAdjustedClose=true'

    @property
    def export_folder(self) -> str:
        return f'audit/yahoo/sp/{self.run_datetime}'

    @property
    def export_filepath(self) -> str:
        return f'{self.export_folder}/sp_{self.run_datetime}.csv'

    def execute(self):
        utils.create_directory(self.export_folder)
        response = requests.get(self.call)
        df = pd.read_csv(io.StringIO(response.text))
        df.to_csv(self.export_filepath)


if __name__ == '__main__':
    YahooSPCaller().execute()
