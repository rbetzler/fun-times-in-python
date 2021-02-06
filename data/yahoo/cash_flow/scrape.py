from data.yahoo import scrape
from utilities import yahoo_utils


class YahooCashFlowQuarterly(scrape.YahooScraper):
    @property
    def yahoo_feed(self) -> str:
        return 'cash_flow_quarterly'

    @property
    def yahoo_url(self) -> str:
        return yahoo_utils.CASH_FLOW_QUARTERLY


class YahooCashFlowAnnual(scrape.YahooScraper):
    @property
    def yahoo_feed(self) -> str:
        return 'cash_flow_annual'

    @property
    def yahoo_url(self) -> str:
        return yahoo_utils.CASH_FLOW_ANNUAL


if __name__ == '__main__':
    batch_size = 5000
    n_batches = 2
    for batch in range(1, n_batches):
        lower_bound = (batch - 1) * batch_size
        print('Beginning Batch: ' + str(batch))
        YahooCashFlowQuarterly(lower_bound=lower_bound, batch_size=batch_size).execute()
        print('Completed Batch: ' + str(batch))

    batch_size = 5000
    n_batches = 2
    for batch in range(1, n_batches):
        lower_bound = (batch - 1) * batch_size
        print('Beginning Batch: ' + str(batch))
        YahooCashFlowAnnual(lower_bound=lower_bound, batch_size=batch_size).execute()
        print('Completed Batch: ' + str(batch))
