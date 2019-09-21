import pandas as pd
from finance.ingestion import loader


class FileIngestion(loader.FileIngestion):
    @property
    def job_name(self) -> str:
        return 'td_equities_api'

    @property
    def import_directory(self) -> str:
        return 'audit/td_ameritrade/equities'

    @property
    def import_file_prefix(self) -> str:
        return 'td_equities_'

    @property
    def place_batch_file(self) -> bool:
        return True

    @property
    def export_folder(self) -> str:
        return 'audit/batches/td_ameritrade/equities'

    @property
    def table(self) -> str:
        return 'equities'

    @property
    def schema(self) -> str:
        return 'td'

    @property
    def parallel_output(self) -> pd.DataFrame:
        return pd.DataFrame()


if __name__ == '__main__':
    FileIngestion().execute()
