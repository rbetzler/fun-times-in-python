import abc
from data import scraper


class YahooScraper(scraper.Caller, abc.ABC):
    @property
    def job_name(self) -> str:
        return f'YAHOO_{self.yahoo_feed.upper()}'

    @property
    @abc.abstractmethod
    def yahoo_feed(self) -> str:
        """Name of data pull for use in filepaths"""
        pass

    @property
    @abc.abstractmethod
    def yahoo_url(self) -> str:
        """Templated URL string to call"""
        pass

    @property
    def calls_query(self) -> str:
        query = r'''
            select symbol as ticker
            from dbt.tickers
            order by 1
            limit {batch_size}
            offset {batch_start}
            '''
        return query.format(batch_size=self.batch_size, batch_start=self.lower_bound)

    def format_calls(self, row) -> tuple:
        key = row.ticker
        request = self.yahoo_url.format(symbol=key)
        return key, request

    @property
    def export_folder(self) -> str:
        return f'audit/yahoo/{self.yahoo_feed}/{self.folder_datetime}/'

    @property
    def export_file_name(self) -> str:
        return f'yahoo_{self.yahoo_feed}_'

    @property
    def n_workers(self) -> int:
        return 10

    @property
    def len_of_pause(self) -> int:
        return 5

    @property
    def export_file_type(self) -> str:
        return ''

    @property
    def write_raw_file(self) -> bool:
        return True
