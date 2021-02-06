from data.yahoo import scrape
from utilities import yahoo_utils


class YahooBalanceSheetQuarterly(scrape.YahooScraper):
    @property
    def yahoo_feed(self) -> str:
        return 'balance_sheet_quarterly'

    @property
    def yahoo_url(self) -> str:
        return yahoo_utils.BALANCE_SHEET_QUARTERLY


class YahooBalanceSheetAnnual(scrape.YahooScraper):
    @property
    def yahoo_feed(self) -> str:
        return 'balance_sheet_annual'

    @property
    def yahoo_url(self) -> str:
        return yahoo_utils.BALANCE_SHEET_ANNUAL


if __name__ == '__main__':
    batch_size = 5000
    n_batches = 2
    for batch in range(1, n_batches):
        lower_bound = (batch - 1) * batch_size
        print('Beginning Batch: ' + str(batch))
        YahooBalanceSheetQuarterly(lower_bound=lower_bound, batch_size=batch_size).execute()
        print('Completed Batch: ' + str(batch))

    batch_size = 5000
    n_batches = 2
    for batch in range(1, n_batches):
        lower_bound = (batch - 1) * batch_size
        print('Beginning Batch: ' + str(batch))
        YahooBalanceSheetAnnual(lower_bound=lower_bound, batch_size=batch_size).execute()
        print('Completed Batch: ' + str(batch))
