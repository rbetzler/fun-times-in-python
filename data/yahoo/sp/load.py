from data import loader


class SPIndexLoader(loader.FileIngestion):
    """
    Load SP 500 Daily Performance CSVs

    Manually pulled from Yahoo Finance
    https://finance.yahoo.com/quote/%5EGSPC/history?period1=-1325635200&period2=1581206400&interval=1d&filter=history&frequency=1d
    """
    @property
    def job_name(self) -> str:
        return 'fred_series_searches'

    @property
    def import_directory(self) -> str:
        return 'audit/yahoo/sp'

    @property
    def import_file_prefix(self) -> str:
        return 'sp500_'

    @property
    def table(self) -> str:
        return 'sp_index'

    @property
    def schema(self) -> str:
        return 'yahoo'

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

    @property
    def load_to_db(self) -> bool:
        return True


if __name__ == '__main__':
    SPIndexLoader().execute()
