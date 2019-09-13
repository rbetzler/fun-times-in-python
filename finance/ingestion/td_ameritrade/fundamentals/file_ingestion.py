import pandas as pd
from finance.ingestion import file_ingestion


class FileIngestion(file_ingestion.FileIngestion):
    @property
    def job_name(self) -> str:
        return 'td_fundamentals_api'

    @property
    def import_directory(self) -> str:
        return 'audit/processed/td_ameritrade/fundamentals'

    @property
    def import_file_prefix(self) -> str:
        return 'td_fundamentals_'

    @property
    def place_batch_file(self) -> bool:
        return True

    @property
    def export_folder(self) -> str:
        return 'audit/batches/td_ameritrade/fundamentals'

    @property
    def export_file_separator(self) -> str:
        return ';'

    @property
    def table(self) -> str:
        return 'fundamentals'

    @property
    def schema(self) -> str:
        return 'td'

    @property
    def parallel_output(self) -> pd.DataFrame:
        return pd.DataFrame()

    def clean_df(self, df) -> pd.DataFrame:
        blank_to_null_cols = ['dividend_date', 'dividend_pay_date']
        for blank_to_null_col in blank_to_null_cols:
            df.loc[df[blank_to_null_col] == ' ', blank_to_null_col] = ''
        return df


if __name__ == '__main__':
    FileIngestion().execute()
