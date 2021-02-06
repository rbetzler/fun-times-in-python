from data.yahoo import scrape


class YahooOptions(scrape.YahooScraper):

    @property
    def yahoo_feed(self) -> str:
        return 'options'

    @property
    def yahoo_url(self) -> None:
        """Not used for yahoo options job"""
        return None

    def format_calls(self, row) -> tuple:
        key = row.ticker
        request = f'https://query2.finance.yahoo.com/v8/finance/chart/{key}?formatted=true&crumb=3eIGSD3T5Ul&lang=en-US&region=US&period1=34450&period2=153145400400&interval=1d&events=div%7Csplit&corsDomain=finance.yahoo.com'
        return key, request


if __name__ == '__main__':
    batch_size = 5000
    n_batches = 2
    for batch in range(1, n_batches):
        lower_bound = (batch - 1) * batch_size
        print('Beginning Batch: ' + str(batch))
        YahooOptions(lower_bound=lower_bound, batch_size=batch_size).execute()
        print('Completed Batch: ' + str(batch))
