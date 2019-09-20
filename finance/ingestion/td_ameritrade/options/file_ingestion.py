import pandas as pd
from finance.ingestion import file_ingestion


class FileIngestion(file_ingestion.FileIngestion):
    @property
    def job_name(self) -> str:
        return 'td_options_api'

    @property
    def import_directory(self) -> str:
        return 'audit/td_ameritrade/options'

    @property
    def import_file_prefix(self) -> str:
        return 'td_'

    @property
    def place_batch_file(self) -> bool:
        return True

    @property
    def export_folder(self) -> str:
        return 'audit/batches/td_ameritrade/options'

    @property
    def table(self) -> str:
        return 'options'

    @property
    def schema(self) -> str:
        return 'td'

    @property
    def parallel_output(self) -> pd.DataFrame:
        return pd.DataFrame()


if __name__ == '__main__':
    FileIngestion().execute()
