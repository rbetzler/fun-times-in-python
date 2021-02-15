from data import reporter


class ListedStocksReport(reporter.Reporter):

    @property
    def query(self) -> str:
        query = '''
            select distinct symbol
            from dbt.tickers
            order by 1
            '''
        return query

    @property
    def export_folder(self) -> str:
        return 'listed_stocks'

    @property
    def export_file_name(self) -> str:
        return 'listed_stocks'

    @property
    def export_file_path(self) -> str:
        return f'{reporter.REPORT_FOLDER_PREFIX}/{self.export_folder}/{self.export_file_name}{self.export_file_type}'


if __name__ == '__main__':
    ListedStocksReport().execute()
