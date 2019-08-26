import os
import abc
import datetime
import pandas as pd
from sqlalchemy import create_engine
from scripts.utilities import db_utilities


class FileIngestion(abc.ABC):
    def __init__(self,
                 start_date=datetime.datetime.now().date().strftime('%Y-%m-%d'),
                 end_date=datetime.datetime.now().date().strftime('%Y-%m-%d')):
        self.db_connection = db_utilities.DW_STOCKS
        self.start_date = start_date
        self.end_date = end_date

    @property
    def run_date(self) -> str:
        return datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S')

    @property
    def place_raw_file(self) -> bool:
        return False

    @property
    def place_batch_file(self) -> bool:
        return False

    @property
    def batch_name(self) -> str:
        return 'batch'

    @property
    def place_with_index(self) -> bool:
        return False

    @property
    def import_directory(self) -> str:
        return ''

    @property
    def import_file_prefix(self) -> str:
        return ''

    @property
    def import_file_date_format(self) -> str:
        return '%Y%m%d%H%M%S'

    @property
    def export_folder(self) -> str:
        return ''

    @property
    def export_file_name(self) -> str:
        return ''

    @property
    def export_file_type(self) -> str:
        return '.csv'

    def export_file_path(self, api) -> str:
        file_path = self.export_folder \
                    + self.export_file_name \
                    + api + '_' \
                    + self.run_date \
                    + self.export_file_type
        return file_path

    @property
    def load_to_db(self) -> bool:
        return False

    @property
    def table(self) -> str:
        return ''

    @property
    def schema(self) -> str:
        return ''

    @property
    def db_engine(self) -> str:
        return create_engine(self.db_connection)

    @property
    def append_to_table(self) -> str:
        return 'append'

    @property
    def index(self) -> bool:
        return False

    @property
    def data_format(self) -> pd.DataFrame:
        return pd.DataFrame()

    def parse(self) -> pd.DataFrame:
        pass

    def execute(self):
        files = os.listdir(self.import_directory)
        files = pd.DataFrame(files, columns=['file_names'])
        # files = files['file_names'].str.find(self.import_file_prefix)
        files['file_date'] = files['file_names'].str.rpartition('_')[2].str.partition('.csv')[0]
        files['file_date'] = pd.to_datetime(files['file_date'], format=self.import_file_date_format)

        df = self.data_format
        for file in files:
            raw = pd.read_csv(file)
            df = pd.concat([df, raw])

        if self.place_batch_file:
            df.to_csv(self.export_file_path(self.batch_name), index=self.place_with_index)

        if self.load_to_db:
            df.to_sql(
                self.table,
                self.db_engine,
                schema=self.schema,
                if_exists=self.append_to_table,
                index=self.index)
