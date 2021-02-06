from data import loader


class SPIndexLoader(loader.FileIngestion):

    @property
    def job_name(self) -> str:
        return 'YAHOO_SP_500'

    @property
    def directory(self) -> str:
        return 'yahoo/sp'

    @property
    def import_file_prefix(self) -> str:
        return 'sp_'

    @property
    def vacuum_analyze(self) -> bool:
        return False

    @property
    def schema(self) -> str:
        return 'yahoo'

    @property
    def table(self) -> str:
        return 'sp_index'

    @property
    def column_mapping(self) -> dict:
        mapping = {
            'Date': 'market_datetime',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }
        return mapping

    def clean_df(self, df):
        df['index_name'] = 'sp_500'
        return df


if __name__ == '__main__':
    SPIndexLoader().execute()
