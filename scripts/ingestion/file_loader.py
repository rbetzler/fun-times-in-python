import abc
import datetime
import pandas as pd
from sqlalchemy import create_engine
from scripts.utilities import db_utilities


class FileLoader(abc.ABC):
    def __init__(self,
                 run_date=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M'),
                 start_date=datetime.datetime.now().date().strftime('%Y-%m-%d'),
                 end_date=datetime.datetime.now().date().strftime('%Y-%m-%d')):
        self.db_connection = db_utilities.DW_STOCKS
        self.run_date = run_date
        self.start_date = start_date
        self.end_date = end_date

    @property
    def place_raw_file(self) -> bool:
        return False

    @property
    def place_with_index(self) -> bool:
        return False

    @property
    def input_folder(self) -> str:
        return ''

    @property
    def input_file_name(self) -> str:
        return ''

    @property
    def input_file_type(self) -> str:
        return '.csv'

    @property
    def input_file_path(self) -> str:
        file_path = 'audit/raw' \
                    + self.input_folder \
                    + self.input_file_name \
                    + self.run_date \
                    + self.input_file_type
        return file_path

    @property
    def export_folder(self) -> str:
        return ''

    @property
    def export_file_name(self) -> str:
        return ''

    @property
    def export_file_type(self) -> str:
        return '.csv'

    @property
    def export_file_path(self) -> str:
        file_path = 'audit/processed' \
                    + self.export_folder \
                    + self.export_file_name \
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
    def column_mapping(self) -> dict:
        return {}

    def parse(self, file) -> pd.DataFrame:
        pass

    def execute(self):
        file = pd.read_csv(self.input_file_path)
        df = self.parse(file)

        if self.place_raw_file:
            df.to_csv(self.export_file_path, index=self.place_with_index)

        if self.load_to_db:
            if 'dw_created_at' not in df:
                df['dw_created_at'] = datetime.datetime.now().utcnow().strftime("%m/%d/%Y %H:%M:%S")
            df.to_sql(
                self.table,
                self.db_engine,
                schema=self.schema,
                if_exists=self.append_to_table,
                index=self.index)
