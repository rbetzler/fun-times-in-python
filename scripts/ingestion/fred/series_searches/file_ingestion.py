import pandas as pd
from scripts.ingestion import file_ingestion


class FREDSeriesFileIngestion(file_ingestion.FileIngestion):
    @property
    def job_name(self) -> str:
        return 'fred_series_searches'

    @property
    def import_directory(self) -> str:
        return 'audit/processed/fred/series_searches'

    @property
    def import_file_prefix(self) -> str:
        return 'fred_series_searches_'

    @property
    def table(self) -> str:
        return 'series'

    @property
    def schema(self) -> str:
        return 'fred'

    @property
    def load_to_db(self) -> bool:
        return True


if __name__ == '__main__':
    FREDSeriesFileIngestion().execute()
