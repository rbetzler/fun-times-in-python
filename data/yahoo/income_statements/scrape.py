from data.yahoo import scrape
from utilities import yahoo_utils


class YahooIncomeStatementsQuarterly(scrape.YahooScraper):
    @property
    def yahoo_feed(self) -> str:
        return 'income_statements_quarterly'

    @property
    def yahoo_url(self) -> str:
        return yahoo_utils.INCOME_STATEMENT_QUARTERLY


class YahooIncomeStatementsAnnual(scrape.YahooScraper):
    @property
    def yahoo_feed(self) -> str:
        return 'income_statements_annual'

    @property
    def yahoo_url(self) -> str:
        return yahoo_utils.INCOME_STATEMENT_ANNUAL


if __name__ == '__main__':
    batch_size = 5000
    n_batches = 2
    for batch in range(1, n_batches):
        lower_bound = (batch - 1) * batch_size
        print('Beginning Batch: ' + str(batch))
        YahooIncomeStatementsQuarterly(lower_bound=lower_bound, batch_size=batch_size).execute()
        print('Completed Batch: ' + str(batch))

    batch_size = 5000
    n_batches = 2
    for batch in range(1, n_batches):
        lower_bound = (batch - 1) * batch_size
        print('Beginning Batch: ' + str(batch))
        YahooIncomeStatementsAnnual(lower_bound=lower_bound, batch_size=batch_size).execute()
        print('Completed Batch: ' + str(batch))
