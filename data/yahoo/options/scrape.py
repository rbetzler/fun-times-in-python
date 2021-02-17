import requests
import time
from data.yahoo import scrape


class YahooOptions(scrape.YahooScraper):

    @property
    def yahoo_feed(self) -> str:
        return 'options'

    @property
    def yahoo_url(self) -> None:
        """Not used for yahoo options job"""
        return None

    def format_calls(self, row, expiration_date: str=None) -> tuple:
        key = row if isinstance(row, str) else row.ticker
        date_param = f'&date={expiration_date}' if expiration_date else ''
        request = f'https://query2.finance.yahoo.com/v7/finance/options/{key}?formatted=true&lang=en-US&region=US{date_param}&corsDomain=finance.yahoo.com'
        return key, request

    @staticmethod
    def _get(call: str, export_file_path: str) -> dict:
        response = requests.get(call)
        output = response.json()
        file = open(export_file_path, 'w')
        file.write(str(output))
        return output

    def get(
            self,
            key: str,
            call: str,
    ):
        """Get data from an api/url, either save the raw file or parse it and write to a csv"""
        export_file_path = self.export_file_path(f'{key}_0')
        output = self._get(call, export_file_path)

        chain = output.get('optionChain')
        underlying = chain.get('result')[0]
        expiration_dates = underlying.get('expirationDates')
        for expiration_date in expiration_dates:
            _, call = self.format_calls(row=key, expiration_date=expiration_date)
            export_file_path = self.export_file_path(f'{key}_{expiration_date}')
            self._get(call, export_file_path)
            time.sleep(self.len_of_pause)


if __name__ == '__main__':
    batch_size = 5000
    n_batches = 2
    for batch in range(1, n_batches):
        lower_bound = (batch - 1) * batch_size
        print('Beginning Batch: ' + str(batch))
        YahooOptions(lower_bound=lower_bound, batch_size=batch_size).execute()
        print('Completed Batch: ' + str(batch))
